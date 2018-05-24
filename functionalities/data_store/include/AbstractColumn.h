#ifndef PROTODB_COLUMNSTORAGE_H
#define PROTODB_COLUMNSTORAGE_H

#include <memory>

/**
 * Defining a shared pointer for this class
 */
class ColumnStorage;
typedef std::shared_ptr<ColumnStorage> ColumnStoragePtr;

/**
 * This is a class that will store a column of data
 * I plan on supporting integers and matrices
 */
class ColumnStorage {

};

#endif //PROTODB_COLUMNSTORAGE_H
