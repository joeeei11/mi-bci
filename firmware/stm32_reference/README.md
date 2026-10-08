# STM32 reference command endpoint

Connect the host's USB CDC, a Bluetooth serial module, and a Wi-Fi TCP-to-UART
module to three independent byte streams. Feed each complete LF-terminated line
to `stm32_control_line`. Call `stm32_control_tick` from a 10 ms timer. Implement
`stm32_control_apply` so STOP disables the motor/actuator enable pin before it
returns. Call `stm32_control_reset` at boot and on every link disconnect.

The wire protocol is ASCII. Host sends `CMD <sequence> LEFT|RIGHT|STOP\n`.
Firmware replies `ACK <sequence> <command>\n` after applying the command, or
`ERR <sequence> INVALID\n`. The sequence is a decimal 32-bit unsigned integer.
Repeating the same sequence and command replies with the same ACK without
reapplying the output. Reusing a sequence for another command is rejected.
Each transport has its own parser and sequence cache. A valid LEFT or RIGHT
expires after 500 ms without a fresh command; STOP is then applied locally.

The desktop requires the exact matching ACK within 300 ms by default. Failure
to receive it is an unconfirmed stop, even when a STOP frame was transmitted.
The gateway cannot prove the actuator is safe after a disconnected link; check
the physical enable circuit before resetting the session. Use a normally-off
output driver and independent hardware watchdog for actual equipment.

This is a HAL integration reference, not a board-specific flash image. USB CDC,
Bluetooth UART, and Wi-Fi module callbacks must each invoke the parser and send
the returned response over the same link. Pin mapping, voltage levels, module
configuration, and real actuator validation are hardware-specific release work.
