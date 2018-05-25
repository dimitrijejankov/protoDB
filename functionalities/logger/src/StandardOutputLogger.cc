#include <iostream>
#include "StandardOutputLogger.h"

StandardOutputLogger::StandardOutputLogger(const std::string &name) : AbstractLogger(name) {}

void StandardOutputLogger::output(const std::string &text) {
  std::cout << text;
}
