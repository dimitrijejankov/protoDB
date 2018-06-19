//
// Created by dimitrije on 5/24/18.
//

#ifndef PROTODB_LOGGER_H
#define PROTODB_LOGGER_H

#include <string>
#include <memory>
#include <AbstractFunctionality.h>
#include <sstream>

/**
 * The abstract pointer
 */
class AbstractLogger;
typedef std::shared_ptr<AbstractLogger> AbstractLoggerPtr;

/**
 * Is the base class for every logger
 */
class AbstractLogger : public AbstractFunctionality {
public:

  /**
   * Just the default destructor
   */
  ~AbstractLogger() override = default;

  /**
   * The constructor of the logger each logger must have a name
   * @param name - the name of the logger
   */
  explicit AbstractLogger(const std::string &name);

  /**
   * Logs the some information that might be useful to have
   * @param text - the information
   */
  AbstractLogger& info();

  /**
   * Logs a warning about something that happened in the system
   * @param text - the text of the warning
   */
  AbstractLogger& warn();

  /**
   * Logs the error about something that happened in the system
   * @param text - the text of the error
   */
  AbstractLogger& error();

  /**
   * Ends the line of the logger
   * @return the end line string
   */
  static const std::string endl;

  /**
   * Outputs a string to the logger
   * @param text - the text
   * @return - the logger again
   */
  AbstractLogger& operator<<(const std::string &text);

  /**
   * Outputs a const char* to the logger
   * @param text the const char* we provided
   * @return the logger again
   */
  AbstractLogger& operator<<(const char *text);

  /**
   * Outputs a value on which we will call std::to_string to convert it into a string
   * @tparam T the type of the value we want to convert to string
   * @param value - the value
   * @return the logger again
   */
  template<typename T>
  AbstractLogger& operator<<(const T &value) {

    std::stringstream stream;
    stream << value;

    // outputs the text to the logger
    output(stream.str());

    // returns the instance back
    return *this;
  }

  /**
   * Returns the type of the logger
   * @return the type
   */
  FunctionalityType getType() override;

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
