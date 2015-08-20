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
#include <variant/variant.hpp>
#include <veles/logger.h>  // NOLINT(*)

template<typename... Types>
using variant = mapbox::util::variant<Types...>;

namespace veles {

class NumpyArrayReference {
 public:
  explicit NumpyArrayReference(const std::string& file_name)
      : file_name_(file_name) {}

  const std::string& file_name() const noexcept { return file_name_; }

 private:
  std::string file_name_;
};

using Property = variant<bool, int, float, std::string, NumpyArrayReference>;

class UnitDefinition {
 public:
  UnitDefinition(const std::string& name, const std::string& uuid);

  std::string name() const noexcept { return name_; }
  const uint8_t* uuid() const noexcept { return uuid_; }
  std::string uuid_str() const noexcept;
  const std::vector<std::shared_ptr<UnitDefinition>>&
  links() const noexcept {
    return links_;
  }
  template <class T>
  const T& get(const std::string& key) const;
  template <class T>
  T& get(const std::string& key);
  template <class T>
  void set(const std::string& key, const T& value);
  std::vector<std::string> PropertyNames() const noexcept;
  void Link(std::shared_ptr<UnitDefinition> def);

 private:
  std::string name_;
  uint8_t uuid_[16];
  std::vector<std::shared_ptr<UnitDefinition>> links_;
  std::unordered_map<std::string, Property> props_;
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
const T& UnitDefinition::get(const std::string& key) const {
  auto val = props_.find(key)->second;
  return mapbox::util::get<T>(val);
}

template <class T>
T& UnitDefinition::get(const std::string& key) {
  return mapbox::util::get<T>(props_[key]);
}

template <class T>
void UnitDefinition::set(const std::string& key, const T& value) {
  props_[key] = value;
}

}  // namespace veles
#endif  // MAIN_FILE_LOADER_H_
