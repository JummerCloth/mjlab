"""Mjlab play viewer based on Viser with simulation controls.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import viser
from typing_extensions import override

from mjlab.sim.sim import Simulation
from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel
from mjlab.viewer.viser.reward_plotter import ViserRewardPlotter
from mjlab.viewer.viser.scene import ViserMujocoScene


class ViserPlayViewer(BaseViewer):
  """Interactive Viser-based viewer with playback controls."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ) -> None:
    super().__init__(env, policy, frame_rate, verbosity)
    self._reward_plotter: ViserRewardPlotter | None = None

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    self._server = viser.ViserServer(label="mjlab")
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._counter = 0
    self._needs_update = False

    # Create ViserMujocoScene for all 3D visualization (with debug visualization enabled).
    self._scene = ViserMujocoScene.create(
      server=self._server,
      mj_model=sim.mj_model,
      num_envs=self.env.num_envs,
    )

    self._scene.env_idx = self.cfg.env_idx
    self._scene.debug_visualization_enabled = (
      True  # Enable debug visualization by default
    )

    # Create tab group.
    tabs = self._server.gui.add_tab_group()

    # Main tab with simulation controls and display settings.
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      # Status display.
      with self._server.gui.add_folder("Info"):
        self._status_html = self._server.gui.add_html("")

      # Simulation controls.
      with self._server.gui.add_folder("Simulation"):
        # Play/Pause button.
        self._pause_button = self._server.gui.add_button(
          "Play" if self._is_paused else "Pause",
          icon=viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE,
        )

        @self._pause_button.on_click
        def _(_) -> None:
          self.toggle_pause()
          self._pause_button.label = "Play" if self._is_paused else "Pause"
          self._pause_button.icon = (
            viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
          )
          self._update_status_display()
          self._needs_update = True

        # Reset button.
        reset_button = self._server.gui.add_button("Reset Environment")

        @reset_button.on_click
        def _(_) -> None:
          self.reset_environment()
          self._update_status_display()
          self._needs_update = True

        # Speed controls.
        speed_buttons = self._server.gui.add_button_group(
          "Speed",
          options=["Slower", "Faster"],
        )

        @speed_buttons.on_click
        def _(event) -> None:
          if event.target.value == "Slower":
            self.decrease_speed()
          else:
            self.increase_speed()
          self._update_status_display()

      # Add standard visualization options from ViserMujocoScene (Environment, Visualization, Contacts, Camera Tracking, Debug Visualization).
      self._scene.create_visualization_gui(
        camera_distance=self.cfg.distance,
        camera_azimuth=self.cfg.azimuth,
        camera_elevation=self.cfg.elevation,
      )

      # Command visualization and manual control for velocity tasks
      if hasattr(self.env.unwrapped, "command_manager"):
        cmd_manager = self.env.unwrapped.command_manager
        if "twist" in cmd_manager._terms:
          with self._server.gui.add_folder("Commands"):
            # Display current commands as text
            self._command_display = self._server.gui.add_html(
              "<div style='font-size: 0.9em; line-height: 1.5; padding: 0.5em;'>"
              "<strong>Current Commands:</strong><br/>"
              "Waiting for first update..."
              "</div>"
            )

            # Manual control mode
            self._manual_control_cb = self._server.gui.add_checkbox(
              "Manual Control",
              initial_value=False,
              hint="Override automatic commands with manual sliders",
            )

            # Manual control sliders
            self._manual_forward_slider = self._server.gui.add_slider(
              "Forward (m/s)",
              min=-0.5,
              max=0.5,
              step=0.05,
              initial_value=0.0,
              disabled=True,
            )
            self._manual_lateral_slider = self._server.gui.add_slider(
              "Lateral (m/s)",
              min=-0.3,
              max=0.3,
              step=0.05,
              initial_value=0.0,
              disabled=True,
            )
            self._manual_turn_slider = self._server.gui.add_slider(
              "Turn (rad/s)",
              min=-2.0,
              max=2.0,
              step=0.1,
              initial_value=0.0,
              disabled=True,
            )

            @self._manual_control_cb.on_update
            def _(_) -> None:
              # Enable/disable sliders based on manual control mode
              is_manual = self._manual_control_cb.value
              self._manual_forward_slider.disabled = not is_manual
              self._manual_lateral_slider.disabled = not is_manual
              self._manual_turn_slider.disabled = not is_manual

    self._prev_env_idx = self._scene.env_idx

    # Reward plots tab.
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        # Get reward term names and create reward plotter.
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._scene.env_idx
          )
        ]
        self._reward_plotter = ViserRewardPlotter(self._server, term_names)

    # Geom groups tab.
    self._scene.create_geom_groups_gui(tabs)

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    
    # Apply manual commands if enabled
    self._apply_manual_commands()
    
    self._counter += 1
    if self._counter % 10 == 0:
      self._update_status_display()
      self._update_command_display()
      
      if self._scene.env_idx != self._prev_env_idx:
        self._prev_env_idx = self._scene.env_idx
        if self._reward_plotter:
          self._reward_plotter.clear_histories()
        # Clear debug visualizations when switching environments
        if self._scene.debug_visualization_enabled:
          self._scene.clear_debug_all()

      if self._reward_plotter is not None and not self._is_paused:
        terms = list(
          self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._scene.env_idx
          )
        )
        self._reward_plotter.update(terms)

    # Update debug visualizations if enabled
    if self._scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self._scene.clear()  # Clear queued arrows from previous frame
      self.env.unwrapped.update_visualizers(self._scene)

    if self._counter % 2 != 0:
      return
    if self._is_paused and not self._needs_update and not self._scene.needs_update:
      return

    def update_scene() -> None:
      with self._server.atomic():
        self._scene.update(sim.wp_data)
        self._server.flush()

    self._threadpool.submit(update_scene)
    self._needs_update = False
    self._scene.needs_update = False

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    pass

  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward histories."""
    super().reset_environment()
    if self._reward_plotter:
      self._reward_plotter.clear_histories()

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self._reward_plotter:
      self._reward_plotter.cleanup()
    self._threadpool.shutdown(wait=True)
    self._server.stop()

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed.

  def _apply_manual_commands(self) -> None:
    """Apply manual commands if manual control is enabled."""
    if not hasattr(self, "_manual_control_cb") or not self._manual_control_cb.value:
      return

    if not hasattr(self.env.unwrapped, "command_manager"):
      return

    cmd_manager = self.env.unwrapped.command_manager
    if "twist" not in cmd_manager._terms:
      return

    cmd_term = cmd_manager.get_term("twist")
    if cmd_term is None:
      return

    # Set manual commands for all environments
    cmd_term.vel_command_b[:, 0] = self._manual_forward_slider.value
    cmd_term.vel_command_b[:, 1] = self._manual_lateral_slider.value
    cmd_term.vel_command_b[:, 2] = self._manual_turn_slider.value

  def _update_command_display(self) -> None:
    """Update the command display with current velocity commands."""
    if not hasattr(self, "_command_display"):
      return

    if not hasattr(self.env.unwrapped, "command_manager"):
      return

    cmd_manager = self.env.unwrapped.command_manager
    if "twist" not in cmd_manager._terms:
      return

    cmd_term = cmd_manager.get_term("twist")
    if cmd_term is None:
      return

    # Get command for the currently selected environment
    cmd = cmd_term.vel_command_b[self._scene.env_idx].cpu().numpy()
    forward = cmd[0]
    lateral = cmd[1]
    turn = cmd[2]

    # Get actual velocities for comparison
    from mjlab.entity import Entity
    robot: Entity = self.env.unwrapped.scene[cmd_term.cfg.asset_name]
    actual_vel = robot.data.root_link_lin_vel_b[self._scene.env_idx].cpu().numpy()
    actual_ang = robot.data.root_link_ang_vel_b[self._scene.env_idx].cpu().numpy()

    # Format the display
    mode_str = (
      "<span style='color: #ff6b6b; font-weight: bold;'>MANUAL</span>"
      if hasattr(self, "_manual_control_cb") and self._manual_control_cb.value
      else "<span style='color: #51cf66;'>AUTO</span>"
    )

    self._command_display.content = f"""
      <div style='font-size: 0.85em; line-height: 1.6; padding: 0.5em; 
                  background: #f8f9fa; border-radius: 4px; 
                  font-family: monospace;'>
        <strong>Commands (Env {self._scene.env_idx})</strong> [{mode_str}]<br/>
        <div style='margin-top: 0.3em;'>
          <strong>Commanded:</strong><br/>
          &nbsp;&nbsp;Forward: {forward:+.3f} m/s<br/>
          &nbsp;&nbsp;Lateral: {lateral:+.3f} m/s<br/>
          &nbsp;&nbsp;Turn: {turn:+.3f} rad/s<br/>
        </div>
        <div style='margin-top: 0.3em;'>
          <strong>Actual:</strong><br/>
          &nbsp;&nbsp;Forward: {actual_vel[0]:+.3f} m/s<br/>
          &nbsp;&nbsp;Lateral: {actual_vel[1]:+.3f} m/s<br/>
          &nbsp;&nbsp;Turn: {actual_ang[2]:+.3f} rad/s<br/>
        </div>
      </div>
    """

  def _update_status_display(self) -> None:
    """Update the HTML status display."""
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong> {"Paused" if self._is_paused else "Running"}<br/>
        <strong>Steps:</strong> {self._step_count}<br/>
        <strong>Speed:</strong> {self._time_multiplier:.0%}
      </div>
      """
