#include "AbstractLogger.h"

AbstractLogger::AbstractLogger(const std::string &name) : name(name) {}

void AbstractLogger::info(const std::string &text) {
  output("[INFO][" + name  + "] : " + text);
}

void AbstractLogger::warn(const std::string &text) {
  output("[WARNING][" + name  + "] : " + text);
}

void AbstractLogger::error(const std::string &text) {
  output("[ERROR][" + name  + "] : " + text);
}

FunctionalityType AbstractLogger::getType() {
  return LOGGER;
}
