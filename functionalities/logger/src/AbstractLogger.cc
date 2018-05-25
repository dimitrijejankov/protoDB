#include "AbstractLogger.h"

AbstractLogger::AbstractLogger(const std::string &name) : name(name) {}

AbstractLogger& AbstractLogger::info() {

  // starts logging info
  output("[INFO][" + name  + "] : ");

  // returns a reference to this object
  return *this;
}

AbstractLogger& AbstractLogger::warn() {

  // starts logging warning
  output("[WARNING][" + name  + "] : ");

  // returns a reference to this object
  return *this;
}

AbstractLogger& AbstractLogger::error() {

  // starts logging the error
  output("[ERROR][" + name  + "] : ");

  // returns a reference to this object
  return *this;
}

FunctionalityType AbstractLogger::getType() {
  return LOGGER;
}

AbstractLogger &AbstractLogger::operator<<(const std::string &text) {
  // outputs the text to the logger
  output(text);

  // returns the instance back
  return *this;
}

AbstractLogger &AbstractLogger::operator<<(const char *text) {

  // outputs the text to the logger
  output(text);

  // returns the instance back
  return *this;
}

const std::string AbstractLogger::endl = "\n";

