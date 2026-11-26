#!/usr/bin/env python3
"""
ADS1292R Hardware Diagnostic & Register Inspector for Raspberry Pi.
Probes SPI0 communication, GPIO states, registers 0x00-0x0B, and DRDY pulses.

Run directly on the Raspberry Pi:
    python3 deploy/inspect_ads1292r.py
"""

import sys
import time

print("===========================================================================")
print("  ADS1292R HARDWARE & SPI DIAGNOSTIC PROBE")
print("===========================================================================\n")

# 1. Check libraries
try:
    import spidev
    print("[1/5] spidev library ..................... PRESENT [OK]")
except ImportError:
    print("[1/5] spidev library ..................... MISSING [FAIL]")
    print("      Run: pip install spidev")
    sys.exit(1)

try:
    import RPi.GPIO as GPIO
    print("[2/5] RPi.GPIO library .................. PRESENT [OK]")
except ImportError:
    print("[2/5] RPi.GPIO library .................. MISSING [FAIL]")
    print("      Run: pip install RPi.GPIO")
    sys.exit(1)

# 2. Pin definitions
DRDY_PIN = 17
START_PIN = 27
PWDN_PIN = 22

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(START_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWDN_PIN, GPIO.OUT, initial=GPIO.HIGH)

drdy_initial = GPIO.input(DRDY_PIN)
print(f"[3/5] GPIO Setup (BCM) .................. COMPLETE [OK]")
print(f"      DRDY Pin 17 initial state: {'HIGH' if drdy_initial else 'LOW'}")

# 3. SPI Bus Init
try:
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1000000
    spi.mode = 0b01
    print("[4/5] SPI0 Bus 0, Device 0 (CE0) ....... OPENED [OK] (Mode 1, 1 MHz)")
except Exception as e:
    print(f"[4/5] SPI0 Bus Init .................... FAILED: {e}")
    sys.exit(1)

# 4. Hardware Reset & SDATAC
print("\n[5/5] Probing ADS1292R Silicon Registers...")
print("      - Pulsing PWDN LOW for 20ms, then HIGH for 200ms...")
GPIO.output(PWDN_PIN, GPIO.LOW)
time.sleep(0.02)
GPIO.output(PWDN_PIN, GPIO.HIGH)
time.sleep(0.20)

def cmd(c):
    spi.xfer2([c])
    time.sleep(0.005)

def read_reg(addr):
    resp = spi.xfer2([0x20 | addr, 0x00, 0x00])
    return resp[2]

def write_reg(addr, val):
    spi.xfer2([0x40 | addr, 0x00, val])
    time.sleep(0.005)

# Send SDATAC twice to ensure continuous mode is cancelled
cmd(0x11)  # SDATAC
time.sleep(0.02)
cmd(0x06)  # RESET
time.sleep(0.10)
cmd(0x11)  # SDATAC
time.sleep(0.02)

REG_NAMES = {
    0x00: "ID",
    0x01: "CONFIG1",
    0x02: "CONFIG2",
    0x03: "LOFF",
    0x04: "CH1SET",
    0x05: "CH2SET",
    0x06: "RLDSENS",
    0x07: "LOFFSENS",
    0x08: "LOFFSTAT",
    0x09: "RESP1",
    0x0A: "RESP2",
    0x0B: "GPIO",
}

print("\n---------------------------------------------------------------------------")
print("  REGISTER DUMP (Address 0x00 to 0x0B)")
print("---------------------------------------------------------------------------")
reg_values = {}
for addr in range(0x0C):
    val = read_reg(addr)
    reg_values[addr] = val
    name = REG_NAMES.get(addr, f"REG_0x{addr:02X}")
    print(f"  Register 0x{addr:02X} ({name:<10}): 0x{val:02X}  (binary: {val:08b})")
print("---------------------------------------------------------------------------")

chip_id = reg_values.get(0x00, 0x00)
print(f"\nCHIP ID EVALUATION: Read 0x{chip_id:02X}")

if chip_id == 0x00:
    print("\n[DIAGNOSIS] CHIP NOT RESPONDING (All Zeros)")
    print("  Root Causes & Checks:")
    print("  1. Power: Verify 3.3V is connected to Pin 1 and GND to Pin 6.")
    print("  2. PWDN / RESET Pin: If Pin 15 (GPIO 22) is not connected to your board's")
    print("     PWDN/RESET header, tie PWDN/RESET directly to 3.3V on the breakout.")
    print("  3. SPI Wiring: Check Pin 19 (MOSI), Pin 21 (MISO), Pin 23 (SCLK), Pin 24 (CE0).")
    print("  4. Chip Select: Ensure ADS1292R CS pin is connected to Pi Pin 24 (CE0).")
elif chip_id == 0xFF:
    print("\n[DIAGNOSIS] MISO HELD HIGH (All Ones)")
    print("  Root Causes & Checks:")
    print("  1. Pin 21 (MISO) is disconnected or floating high.")
    print("  2. ADS1292R DOUT is not wired to Pi Pin 21.")
elif (chip_id & 0x1F) in (0x12, 0x13) or chip_id in (0x53, 0x73, 0x93, 0x52, 0x72):
    chip_name = "ADS1292R" if (chip_id & 0x1F) == 0x13 or chip_id in (0x53, 0x73) else "ADS1292"
    print(f"\n[DIAGNOSIS] SUCCESS! Valid {chip_name} silicon detected (ID=0x{chip_id:02X})")
    
    # Configure and test DRDY pulsing
    print("\nTesting Conversion & DRDY Pulsing...")
    write_reg(0x01, 0x02)  # 500 SPS
    write_reg(0x02, 0xA0)  # Reference ON
    write_reg(0x04, 0x00)  # Ch1 Normal
    write_reg(0x05, 0x00)  # Ch2 Normal
    write_reg(0x06, 0x2C)  # RLD
    cmd(0x08)              # START
    cmd(0x10)              # RDATAC
    GPIO.output(START_PIN, GPIO.HIGH)

    print("Waiting for DRDY transitions...")
    transitions = 0
    t_test_end = time.perf_counter() + 1.0
    last_drdy = GPIO.input(DRDY_PIN)
    while time.perf_counter() < t_test_end:
        curr_drdy = GPIO.input(DRDY_PIN)
        if last_drdy == GPIO.HIGH and curr_drdy == GPIO.LOW:
            transitions += 1
            # Read one frame
            frame = spi.xfer2([0x00] * 9)
        last_drdy = curr_drdy

    print(f"DRDY Falling Edges detected in 1 second: {transitions}")
    if transitions > 100:
        print("[PASS] ADS1292R is actively converting and DRDY is pulsing correctly!")
    else:
        print("[WARN] DRDY transitions were low. Check START pin (Pin 13 / GPIO 27).")

else:
    print(f"\n[DIAGNOSIS] Unexpected ID byte: 0x{chip_id:02X}")
    print("  Verify that the part is a genuine Texas Instruments ADS1292/ADS1292R.")

print("\n===========================================================================")
cmd(0x11)
spi.close()
GPIO.cleanup()
