#include "sfl/client_options.h"
#include "sfl/client_runner.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        return sflclean::run_client(sflclean::parse_client_options(argc, argv));
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()) != "help requested") {
            std::cerr << "configuration error: " << error.what() << "\n\n";
        }
        std::cerr << sflclean::client_usage(argc > 0 ? argv[0] : "sfl_android_client");
        return std::string(error.what()) == "help requested" ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }
}
