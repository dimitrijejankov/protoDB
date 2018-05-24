#ifndef PROTODB_STANDARDOUTPUTLOGGER_H
#define PROTODB_STANDARDOUTPUTLOGGER_H

#include "AbstractLogger.h"
class StandardOutputLogger : public AbstractLogger {

 public:

  /**
   * Creates the logger
   * @param name - the name of the logger
   */
  explicit StandardOutputLogger(const std::string &name);

 protected:

  /**
   * Method that does the outputting
   * @param text - the text we are outputting
   */
  void output(const std::string &text) override;

};

#endif //PROTODB_STANDARDOUTPUTLOGGER_H
