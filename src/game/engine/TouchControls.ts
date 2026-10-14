export interface TouchState {
  moveX: number; // -1 to 1 (strafe left/right)
  moveZ: number; // -1 to 1 (forward/backward)
  isSprinting: boolean;
  lookDeltaX: number; // yaw change
  lookDeltaY: number; // pitch change
}

export class TouchControls {
  public moveVector = { x: 0, z: 0 };
  public isSprinting = false;
  public lookDelta = { x: 0, y: 0 };
  public isTouchDevice = false;

  private joystickTouchId: number | null = null;
  private lookTouchId: number | null = null;

  private joystickOrigin = { x: 0, y: 0 };
  private joystickCurrent = { x: 0, y: 0 };
  private lastLookPos = { x: 0, y: 0 };

  private joystickRadius = 55;

  // Visual callbacks for custom UI rendering if needed
  public onJoystickUpdate?: (active: boolean, origin: { x: number; y: number }, current: { x: number; y: number }) => void;

  constructor() {
    this.isTouchDevice = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
    this.init();
  }

  private init() {
    window.addEventListener('touchstart', this.onTouchStart, { passive: false });
    window.addEventListener('touchmove', this.onTouchMove, { passive: false });
    window.addEventListener('touchend', this.onTouchEnd, { passive: false });
    window.addEventListener('touchcancel', this.onTouchEnd, { passive: false });
  }

  private onTouchStart = (e: TouchEvent) => {
    // Only capture touches that aren't on interactive UI buttons
    for (let i = 0; i < e.changedTouches.length; i++) {
      const touch = e.changedTouches[i];
      const target = touch.target as HTMLElement;

      // Ignore touches on buttons, inputs, links, or modals
      if (target.closest('button, input, textarea, a, [role="button"], .interactive-ui')) {
        continue;
      }

      const screenWidth = window.innerWidth;

      // Left half = Virtual Joystick
      if (touch.clientX < screenWidth * 0.45 && this.joystickTouchId === null) {
        this.joystickTouchId = touch.identifier;
        this.joystickOrigin = { x: touch.clientX, y: touch.clientY };
        this.joystickCurrent = { x: touch.clientX, y: touch.clientY };
        this.onJoystickUpdate?.(true, this.joystickOrigin, this.joystickCurrent);
      }
      // Right half = Camera Look
      else if (touch.clientX >= screenWidth * 0.45 && this.lookTouchId === null) {
        this.lookTouchId = touch.identifier;
        this.lastLookPos = { x: touch.clientX, y: touch.clientY };
      }
    }
  };

  private onTouchMove = (e: TouchEvent) => {
    for (let i = 0; i < e.changedTouches.length; i++) {
      const touch = e.changedTouches[i];

      if (touch.identifier === this.joystickTouchId) {
        const dx = touch.clientX - this.joystickOrigin.x;
        const dy = touch.clientY - this.joystickOrigin.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        const clampedDist = Math.min(dist, this.joystickRadius);
        const angle = Math.atan2(dy, dx);

        this.joystickCurrent = {
          x: this.joystickOrigin.x + Math.cos(angle) * clampedDist,
          y: this.joystickOrigin.y + Math.sin(angle) * clampedDist,
        };

        const normalizedRatio = clampedDist / this.joystickRadius;
        this.moveVector.x = (dx / dist) * normalizedRatio;
        this.moveVector.z = -(dy / dist) * normalizedRatio; // -y is forward

        this.isSprinting = normalizedRatio > 0.85;
        this.onJoystickUpdate?.(true, this.joystickOrigin, this.joystickCurrent);
      } else if (touch.identifier === this.lookTouchId) {
        const dx = touch.clientX - this.lastLookPos.x;
        const dy = touch.clientY - this.lastLookPos.y;

        this.lookDelta.x += dx * 0.0035;
        this.lookDelta.y += dy * 0.0035;

        this.lastLookPos = { x: touch.clientX, y: touch.clientY };
      }
    }
  };

  private onTouchEnd = (e: TouchEvent) => {
    for (let i = 0; i < e.changedTouches.length; i++) {
      const touch = e.changedTouches[i];

      if (touch.identifier === this.joystickTouchId) {
        this.joystickTouchId = null;
        this.moveVector = { x: 0, z: 0 };
        this.isSprinting = false;
        this.onJoystickUpdate?.(false, { x: 0, y: 0 }, { x: 0, y: 0 });
      } else if (touch.identifier === this.lookTouchId) {
        this.lookTouchId = null;
      }
    }
  };

  public consumeLookDelta(): { x: number; y: number } {
    const res = { x: this.lookDelta.x, y: this.lookDelta.y };
    this.lookDelta = { x: 0, y: 0 };
    return res;
  }

  public destroy() {
    window.removeEventListener('touchstart', this.onTouchStart);
    window.removeEventListener('touchmove', this.onTouchMove);
    window.removeEventListener('touchend', this.onTouchEnd);
    window.removeEventListener('touchcancel', this.onTouchEnd);
  }
}
