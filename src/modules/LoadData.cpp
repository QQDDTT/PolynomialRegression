#include "LoadData.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

// 加载数据函数
// 参数：
//   filePath - 数据文件路径
//   X - 特征矩阵
//   y - 标签向量
// 返回值：
//   true 表示加载成功，抛出异常表示加载失败
bool loadData(const std::string& filePath, arma::mat& X, arma::vec& y) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        // 如果文件无法打开，抛出异常并打印错误信息
        std::cerr << "错误：无法打开文件：" << filePath << std::endl;
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::vector<std::vector<double>> data; // 用于临时存储数据
    std::string line;

    // 按行读取文件内容
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;

        // 按逗号分割每一行的数据
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value)); // 将字符串转换为双精度数
        }
        data.push_back(row);
    }
    file.close(); // 关闭文件

    // 检查数据是否为空
    if (data.empty()) {
        std::cerr << "错误：文件中未找到数据。" << std::endl;
        throw std::runtime_error("No data found in file.");
    }

    size_t numFeatures = data[0].size() - 1; // 特征列的数量（最后一列为标签）
    X.set_size(data.size(), numFeatures);   // 设置特征矩阵的大小
    y.set_size(data.size());                // 设置标签向量的大小

    // 填充特征矩阵和标签向量
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < numFeatures; ++j) {
            X(i, j) = data[i][j]; // 填充特征矩阵
        }
        y(i) = data[i][numFeatures]; // 填充标签向量
    }

    // 打印加载成功的信息
    std::cout << "数据加载成功！特征维度：" << X.n_cols
              << "，样本数量：" << X.n_rows << "。" << std::endl;

    return true;
}
