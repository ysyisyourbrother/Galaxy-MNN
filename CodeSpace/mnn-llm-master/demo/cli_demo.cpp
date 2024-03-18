//
//  cli_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

void benchmark(Llm* llm, std::string prompt_file) {
    // 从文件中读取待预测的提示语句
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        // prompt start with '#' will be ignored
        // 以 '#' 开头的行将被忽略
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        prompts.push_back(prompt);
    }
    // 初始化一些计数器和时间变量
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    // 预热模型
    llm->warmup();

    // 对每一个提示语句进行预测和评测
    for (int i = 0; i < prompts.size(); i++) {
        llm->response(prompts[i]);
         // 更新计数器和时间变量
        prompt_len += llm->prompt_len_;
        decode_len += llm->gen_seq_len_;
        prefill_time += llm->prefill_us_;
        decode_time += llm->decode_us_;
        // 重置模型状态，准备处理下一个提示语句
        llm->reset();
    }
    // 计算一些统计指标
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;

     // 打印统计结果
    printf("\n#################################\n");
    printf("prompt tokens num  = %d\n", prompt_len);
    printf("decode tokens num  = %d\n", decode_len);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
}

int main(int argc, const char* argv[]) {
    // 检查命令行参数
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_dir <prompt.txt>" << std::endl;
        return 0;
    }
    // 获取模型目录和待预测的提示语句文件
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    // 创建 Llm 对象并加载模型
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load(model_dir);
    // 根据命令行参数选择运行模式
    if (argc < 3) {
        llm->chat();
    }
    std::string prompt_file = argv[2];
    benchmark(llm.get(), prompt_file);
    return 0;
}
