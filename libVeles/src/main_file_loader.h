/*! @file main_file_loader.h
 *  @brief Declaration of MainFileLoader class.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef MAIN_FILE_LOADER_H_
#define MAIN_FILE_LOADER_H_

#include <istream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <variant/variant.hpp>
#include <veles/logger.h>  // NOLINT(*)

template<typename... Types>
using variant = mapbox::util::variant<Types...>;

namespace veles {

class UnitDefinition {
 public:
  UnitDefinition(const std::string& name, const std::string& uuid);

  std::string name() const noexcept { return name_; }
  const uint8_t* uuid() const noexcept { return uuid_; }
  const std::unordered_set<std::shared_ptr<UnitDefinition>>&
  links() const noexcept {
    return links_;
  }
  template <class T>
  const T& operator[](const std::string& key) const;
  template <class T>
  T& operator[](const std::string& key);
  template <class T>
  void set(const std::string& key, const T& value);
  std::vector<std::string> PropertyNames() const noexcept;
  void Link(std::shared_ptr<UnitDefinition> def);

 private:
  std::string name_;
  uint8_t uuid_[16];
  std::unordered_set<std::shared_ptr<UnitDefinition>> links_;
  std::unordered_map<std::string, variant<bool, int, float, std::string>> props_;
};

class WorkflowDefinition {
 public:
  WorkflowDefinition(const std::string& checksum, const std::string& name,
                     std::shared_ptr<UnitDefinition> start);
  std::string checksum() const noexcept { return checksum_; }
  std::string name() const noexcept { return name_; }
  std::shared_ptr<UnitDefinition> start() const noexcept { return start_; }

 private:
  std::string checksum_;
  std::string name_;
  std::shared_ptr<UnitDefinition> start_;
};

/// Reads and parses the main file with description of the package (e.g.,
/// "contents.json").
class MainFileLoader : protected DefaultLogger<MainFileLoader,
                                               Logger::COLOR_YELLOW> {
 public:
  virtual ~MainFileLoader() = default;

  WorkflowDefinition Load(std::istream* src);
};

template <class T>
const T& UnitDefinition::operator[](const std::string& key) const {
  auto val = props_.find(key)->second;
  return mapbox::util::get<T>(val);
}

template <class T>
T& UnitDefinition::operator[](const std::string& key) {
  return mapbox::util::get<T>(props_[key]);
}

template <class T>
void UnitDefinition::set(const std::string& key, const T& value) {
  props_[key] = value;
}

}  // namespace veles
#endif  // MAIN_FILE_LOADER_H_
