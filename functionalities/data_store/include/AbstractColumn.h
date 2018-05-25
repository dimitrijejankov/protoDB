#ifndef PROTODB_COLUMNSTORAGE_H
#define PROTODB_COLUMNSTORAGE_H

#include <memory>

/**
 * Defining a shared pointer for this class
 */
class AbstractColumn;
typedef std::shared_ptr<AbstractColumn> AbstractColumnPtr;

/**
 * This is a class that will store a column of data
 * I plan on supporting integers and matrices
 */
class AbstractColumn {

public:

  /**
   * Initializes an abstract column with a name
   * @param name - the name of the column
   */
  explicit AbstractColumn(const std::string &name);

protected:

  /**
   * The name of this column
   */
  std::string name;
};

#endif //PROTODB_COLUMNSTORAGE_H
