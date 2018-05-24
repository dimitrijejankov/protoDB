//
// Created by dimitrije on 5/24/18.
//

#ifndef PROTODB_LOGGER_H
#define PROTODB_LOGGER_H

#include <string>
#include <memory>

/**
 * The abstract pointer
 */
class AbstractLogger;
typedef std::shared_ptr<AbstractLogger> AbstractLoggerPtr;

/**
 * Is the base class for every logger
 */
class AbstractLogger {
public:

  /**
   * The constructor of the logger each logger must have a name
   * @param name - the name of the logger
   */
  explicit AbstractLogger(const std::string &name);

  /**
   * Logs the some information that might be useful to have
   * @param text - the information
   */
  void info(const std::string &text);

  /**
   * Logs a warning about something that happened in the system
   * @param text - the text of the warning
   */
  void warn(const std::string &text);

  /**
   * Logs the error about something that happened in the system
   * @param text - the text of the error
   */
  void error(const std::string &text);

protected:

  /**
   * The name of the logger
   */
  std::string name;

  /**
   * Each class that derives from this class provides its own output
   */
  virtual void output(const std::string &text) = 0;

};

#endif //PROTODB_LOGGER_H
