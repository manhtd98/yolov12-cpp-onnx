#include "logger.h"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

void Logger::setLogLevel(LogLevel level) {
    logLevel = level;
}

void Logger::setLogFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex);
    if (logFile.is_open()) {
        logFile.close();
    }
    logFile.open(filename, std::ios::app);
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < logLevel) {
        return;
    }

    std::string levelStr;
    switch (level) {
        case LogLevel::DEBUG:   levelStr = "DEBUG"; break;
        case LogLevel::INFO:    levelStr = "INFO"; break;
        case LogLevel::WARNING: levelStr = "WARNING"; break;
        case LogLevel::ERROR:   levelStr = "ERROR"; break;
    }

    std::string timestamp = getCurrentTime();
    std::stringstream logMessage;
    logMessage << "[" << timestamp << "] [" << levelStr << "] " << message;

    std::lock_guard<std::mutex> lock(mutex);
    if (logFile.is_open()) {
        logFile << logMessage.str() << std::endl;
    }
    std::cout << logMessage.str() << std::endl;
}

void Logger::debug(const std::string& message) { log(LogLevel::DEBUG, message); }
void Logger::info(const std::string& message) { log(LogLevel::INFO, message); }
void Logger::warning(const std::string& message) { log(LogLevel::WARNING, message); }
void Logger::error(const std::string& message) { log(LogLevel::ERROR, message); }

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

std::string Logger::getCurrentTime() {
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}