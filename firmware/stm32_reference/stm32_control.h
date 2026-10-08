#ifndef STM32_CONTROL_H
#define STM32_CONTROL_H

#include <stddef.h>
#include <stdint.h>

typedef enum { STM32_STOP, STM32_LEFT, STM32_RIGHT } stm32_command;
typedef struct {
    uint32_t last_sequence;
    stm32_command last_command;
    uint32_t last_valid_ms;
    uint8_t has_sequence;
} stm32_control;

/* The board integration must make STOP physically disable output. */
void stm32_control_apply(stm32_command command);
void stm32_control_reset(stm32_control *control);
size_t stm32_control_line(stm32_control *control, const char *line, uint32_t now_ms,
                          char *reply, size_t reply_size);
void stm32_control_tick(stm32_control *control, uint32_t now_ms);

#endif
