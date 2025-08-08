#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <string_view>
#include <ranges>
#include <algorithm>
#include <format>
#include <expected>
#include <span>

class CSVParser {
public:
    struct Row {
        std::vector<std::string> columns;
        
        // Convenient accessors for 3-column format
        const std::string& operator[](size_t index) const {
            return index < columns.size() ? columns[index] : empty_string;
        }
        
        size_t size() const { return columns.size(); }
        bool empty() const { return columns.empty(); }
        
        auto begin() const { return columns.begin(); }
        auto end() const { return columns.end(); }
        
    private:
        static inline const std::string empty_string{};
    };
    
    struct ParseResult {
        std::vector<std::string> headers;
        std::vector<Row> rows;
    };
    
    enum class ParseError {
        FileNotFound,
        InvalidFormat,
        EmptyFile
    };

    // Parse from file
    static std::expected<ParseResult, ParseError> parse_file(const std::string& filename, 
                                                           char delimiter = ',', 
                                                           char quote_char = '"') {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return std::unexpected(ParseError::FileNotFound);
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        if (content.empty()) {
            return std::unexpected(ParseError::EmptyFile);
        }
        
        return parse_string(content, delimiter, quote_char);
    }
    
    // Parse from string
    static std::expected<ParseResult, ParseError> parse_string(const std::string& csv_content,
                                                             char delimiter = ',',
                                                             char quote_char = '"') {
        if (csv_content.empty()) {
            return std::unexpected(ParseError::EmptyFile);
        }
        
        ParseResult result;
        auto lines = csv_content 
                   | std::views::split('\n') 
                   | std::views::transform([](auto&& range) {
                       return std::string_view{range.begin(), range.end()};
                   })
                   | std::views::filter([](std::string_view line) {
                       return !line.empty() && line != "\r";
                   });
        
        bool first_line = true;
        for (auto line : lines) {
            // Remove carriage return if present
            if (!line.empty() && line.back() == '\r') {
                line.remove_suffix(1);
            }
            
            auto parsed_row = parse_line(line, delimiter, quote_char);
            
            if (first_line) {
                result.headers = std::move(parsed_row);
                first_line = false;
            } else {
                result.rows.emplace_back(Row{std::move(parsed_row)});
            }
        }
        
        return result;
    }

private:
    static std::vector<std::string> parse_line(std::string_view line, 
                                             char delimiter, 
                                             char quote_char) {
        std::vector<std::string> fields;
        std::string current_field;
        bool in_quotes = false;
        bool quote_encountered = false;
        
        for (size_t i = 0; i < line.size(); ++i) {
            char ch = line[i];
            
            if (ch == quote_char) {
                if (!in_quotes) {
                    in_quotes = true;
                    quote_encountered = true;
                } else if (i + 1 < line.size() && line[i + 1] == quote_char) {
                    // Escaped quote
                    current_field += quote_char;
                    ++i; // Skip next quote
                } else {
                    in_quotes = false;
                }
            } else if (ch == delimiter && !in_quotes) {
                // Trim whitespace only if field wasn't quoted
                if (!quote_encountered) {
                    current_field = trim(current_field);
                }
                fields.push_back(std::move(current_field));
                current_field.clear();
                quote_encountered = false;
            } else {
                current_field += ch;
            }
        }
        
        // Add the last field
        if (!quote_encountered) {
            current_field = trim(current_field);
        }
        fields.push_back(std::move(current_field));
        
        return fields;
    }
    
    static std::string trim(const std::string& str) {
        auto start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        
        auto end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }
};

// Utility function to print parse results
void print_csv_data(const CSVParser::ParseResult& result) {
    // Print headers
    std::cout << "Headers: ";
    for (size_t i = 0; i < result.headers.size(); ++i) {
        std::cout << std::format("\"{}\"", result.headers[i]);
        if (i < result.headers.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";
    
    // Print rows
    for (size_t row_idx = 0; row_idx < result.rows.size(); ++row_idx) {
        const auto& row = result.rows[row_idx];
        std::cout << std::format("Row {}: ", row_idx + 1);
        
        for (size_t col_idx = 0; col_idx < row.size(); ++col_idx) {
            std::cout << std::format("\"{}\"", row[col_idx]);
            if (col_idx < row.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

// Example usage and demonstration
int main() {
    // Example CSV data with headers
    std::string sample_csv = R"(Name,Age,City
"John Doe",30,"New York"
"Jane Smith",25,Los Angeles
"Bob Johnson",35,"Chicago, IL"
"Alice Brown",28,Boston)";
    
    std::cout << "=== Parsing CSV from string ===\n";
    auto result = CSVParser::parse_string(sample_csv);
    
    if (result) {
        std::cout << std::format("Successfully parsed {} rows with {} columns\n", 
                                result->rows.size(), result->headers.size());
        print_csv_data(*result);
        
        // Demonstrate range-based access
        std::cout << "\n=== Range-based iteration ===\n";
        for (const auto& [idx, row] : std::views::enumerate(result->rows)) {
            std::cout << std::format("Row {}: ", idx + 1);
            for (const auto& field : row) {
                std::cout << std::format("[{}] ", field);
            }
            std::cout << "\n";
        }
        
        // Demonstrate filtering with C++23 ranges
        std::cout << "\n=== Filtered results (age > 28) ===\n";
        auto age_header_idx = std::ranges::find(result->headers, "Age") - result->headers.begin();
        
        auto filtered_rows = result->rows 
                           | std::views::filter([age_header_idx](const auto& row) {
                               if (age_header_idx < row.size()) {
                                   try {
                                       return std::stoi(row[age_header_idx]) > 28;
                                   } catch (...) {
                                       return false;
                                   }
                               }
                               return false;
                           });
        
        for (const auto& row : filtered_rows) {
            std::cout << std::format("Name: {}, Age: {}, City: {}\n", 
                                   row[0], row[1], row[2]);
        }
        
    } else {
        std::cout << "Failed to parse CSV\n";
    }
    
    // Example of file parsing (commented out since file may not exist)
    /*
    std::cout << "\n=== Parsing CSV from file ===\n";
    auto file_result = CSVParser::parse_file("data.csv");
    if (file_result) {
        print_csv_data(*file_result);
    } else {
        std::cout << "Could not read file\n";
    }
    */
    
    return 0;
}
