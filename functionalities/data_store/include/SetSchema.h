#ifndef PROTODB_SETSCHEMA_H
#define PROTODB_SETSCHEMA_H

#include <unordered_map>
#include <boost/shared_ptr.hpp>

/**
 * The type of attribute
 */
enum AttributeType {
  INT_TYPE,
  MATRIX_TYPE
};


/**
 * The set schema shared pointer definition
 */
class SetSchema;
typedef std::shared_ptr<SetSchema> SetSchemaPtr;

/**
 * Represents the set schema
 * currently supports only int and matrix types
 */
class SetSchema {
public:

  /**
   * The identifier of the schema
   * @return the schema identifier "databaseName:setName"
   */
  std::string getSchemaIdentifier();

  /**
   * Returns the attributes of this schema
   * @return the attributes
   */
  const std::unordered_map<std::string, AttributeType> &getAttributes() const;

 protected:

  /**
   * The attributes that are in the schema
   */
  std::unordered_map<std::string, AttributeType> attributes;

  /**
   * The name of the set
   */
  std::string setName;

  /**
   * The database name
   */
  std::string databaseName;



};

#endif //PROTODB_SETSCHEMA_H
