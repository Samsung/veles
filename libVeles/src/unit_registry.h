/*! @file unit_registry.h
 *  @brief Defines the unit factory API.
 *  @author markhor
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_UNIT_REGISTRY_H_
#define SRC_UNIT_REGISTRY_H_

#include <memory>
#include <unordered_map>
#include "inc/veles/unit.h"

namespace Veles {

/// @brief Unit factory class.
/// @details Use it the following way:
/// auto unit = UnitFactory::Instance()["All2All"]();
/// @note Meyers singleton ideal for C++11.
class UnitFactory {
  template <class T>
  friend class RegisterUnit;
 public:
  /// @brief Factory function type.
  typedef std::function<std::shared_ptr<Unit>(void)> UnitConstructor;

  /// @brief Map of Unit names to factory functions.
  typedef std::unordered_map<std::string, UnitConstructor> FactoryMap;

  /// @brief Returns the unique instance of UnitFactory class.
  static const UnitFactory& Instance();

  /// @brief Returns the factory function for the requested Unit name.
  /// @param name The name of the Unit.
  UnitConstructor operator[](const std::string& name) const;

  /// @brief Prints the names of registered units to stdout.
  void PrintRegisteredUnits() const;

 private:
  UnitFactory();
  ~UnitFactory();
  UnitFactory(const UnitFactory&) = delete;
  UnitFactory& operator=(const UnitFactory&) = delete;

  static UnitFactory& InstanceRW();

  FactoryMap map_;
};

/// @brief Helper class used to register units. Usually, you do not
/// want to use it explicitly but rather through REGISTER_UNIT macro.
template<class T>
class RegisterUnit {
 public:
  /// @brief This function is called during the execution of static
  /// constructors of global RegisterUnit class instances in each of the
  /// unit's object file (declared in .cc using REGISTER_UNIT macro).
  /// @details Using UnitFactory lazy singleton is safe even during library
  /// load process, thanks to guaranteed local static variables behavior.
  /// @note Never use any global or static variables in your unit's
  /// default constructor! The order of execution of static constructors is
  /// undefined.
  RegisterUnit() {
    T unit;
    UnitFactory::InstanceRW().map_[unit.Name()] = std::make_shared<T>;
  }
};

/// @brief Unit registration utility.
/// @param T The type of your Unit-derived class.
#define REGISTER_UNIT(T) RegisterUnit<T> T##RegistryInstance

}  // namespace Veles

#endif  // SRC_UNIT_REGISTRY_H_
