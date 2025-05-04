#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>

class Logger {
public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogLevel(LogLevel level);
    void setLogFile(const std::string& filename);
    
    void log(LogLevel level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

private:
    Logger() : logLevel(LogLevel::INFO) {}
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string getCurrentTime();

    LogLevel logLevel;
    std::ofstream logFile;
    std::mutex mutex;
};

#endif // LOGGER_H