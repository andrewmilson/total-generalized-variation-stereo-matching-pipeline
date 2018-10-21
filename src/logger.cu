#include <stdio.h>
#include "logger.h"

int updateCount = 0;

void updateMsg(char *message) {
  printf("Step %d: %s...\n", ++updateCount, message);
}
