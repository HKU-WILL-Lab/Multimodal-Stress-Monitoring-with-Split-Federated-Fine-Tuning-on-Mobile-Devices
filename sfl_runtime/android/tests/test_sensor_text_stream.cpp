#include "sfl/sensor_text_stream.h"

#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "sfl_sensor_stream_test.jsonl";
    {
        std::ofstream output(path);
        output << R"({"sensor":[1,2],"token_ids":[10,11,0],"attention_mask":[1,1,0],"loss_mask":[0,1,0]})" << '\n';
        output << R"({"sensor":[3,4],"token_ids":[12,13,14],"attention_mask":[1,1,1],"loss_mask":[0,1,1]})" << '\n';
    }
    sflclean::SensorTextStream stream(path.string(), 0, 1, 2, 3);
    const auto batch = stream.next_batch(2);
    if (batch.batch_size != 2 || batch.sensor_values != std::vector<float>{1, 2, 3, 4} ||
        batch.loss_mask != std::vector<std::int64_t>{0, 1, 0, 0, 1, 1}) {
        std::cerr << "unexpected sensor batch" << std::endl;
        return 1;
    }
    std::filesystem::remove(path);
    std::cout << "SensorTextStream tests passed\n";
    return 0;
}
