#!/usr/bin/env python3
"""
List Output Devices

Lists all available audio output devices with their number of channels and sample rate.
"""
import sounddevice as sd


def list_output_devices():
    """Print all output devices with their channels and sample rates."""
    print("=" * 80)
    print("Available Audio Output Devices")
    print("=" * 80)
    print()
    
    devices = sd.query_devices()
    default_output = sd.default.device[1] if sd.default.device[1] is not None else sd.default.device[0]
    
    output_devices = []
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices.append((i, device))
    
    if not output_devices:
        print("No output devices found.")
        return
    
    for i, device in output_devices:
        marker = " (DEFAULT)" if i == default_output else ""
        print(f"[{i}] {device['name']}{marker}")
        print(f"    Channels: {device['max_output_channels']}")
        print(f"    Sample Rate: {device['default_samplerate']:.0f} Hz")
        print()
    
    print("=" * 80)
    print(f"Total output devices: {len(output_devices)}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        list_output_devices()
    except Exception as e:
        print(f"Error listing devices: {e}")
        import traceback
        traceback.print_exc()



