#include "stm32_control.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

void stm32_control_reset(stm32_control *control) {
    control->has_sequence = 0;
    control->last_valid_ms = 0;
    control->last_command = STM32_STOP;
    stm32_control_apply(STM32_STOP);
}

size_t stm32_control_line(stm32_control *control, const char *line, uint32_t now_ms,
                          char *reply, size_t reply_size) {
    uint32_t sequence;
    char command[6];
    char tail;
    stm32_command parsed;
    int count;
    if (sscanf(line, "CMD %" SCNu32 " %5[A-Z] %c", &sequence, command, &tail) != 2)
        return 0;
    if (strcmp(command, "STOP") == 0) parsed = STM32_STOP;
    else if (strcmp(command, "LEFT") == 0) parsed = STM32_LEFT;
    else if (strcmp(command, "RIGHT") == 0) parsed = STM32_RIGHT;
    else {
        count = snprintf(reply, reply_size, "ERR %" PRIu32 " INVALID\n", sequence);
        return count > 0 && (size_t)count < reply_size ? (size_t)count : 0;
    }
    if (control->has_sequence && (sequence < control->last_sequence ||
        (sequence == control->last_sequence && parsed != control->last_command))) {
        count = snprintf(reply, reply_size, "ERR %" PRIu32 " INVALID\n", sequence);
        return count > 0 && (size_t)count < reply_size ? (size_t)count : 0;
    }
    if (!control->has_sequence || sequence != control->last_sequence) {
        stm32_control_apply(parsed);
        control->last_sequence = sequence;
        control->last_command = parsed;
        control->has_sequence = 1;
    }
    control->last_valid_ms = now_ms;
    count = snprintf(reply, reply_size, "ACK %" PRIu32 " %s\n", sequence, command);
    return count > 0 && (size_t)count < reply_size ? (size_t)count : 0;
}

void stm32_control_tick(stm32_control *control, uint32_t now_ms) {
    if (control->last_command != STM32_STOP && now_ms - control->last_valid_ms > 500) {
        stm32_control_apply(STM32_STOP);
        control->last_command = STM32_STOP;
    }
}
