#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <armadillo>
#include <string>

// 函数：loadData
// 功能：从指定文件路径加载数据，将其分解为特征矩阵和标签向量
// 参数：
//   filePath - 输入，数据文件路径，要求 CSV 格式
//   X - 输出，arma::mat 类型，用于存储特征矩阵
//   y - 输出，arma::vec 类型，用于存储标签向量
// 返回值：
//   如果加载成功，返回 true；如果失败，抛出异常
bool loadData(const std::string& filePath, arma::mat& X, arma::vec& y);

#endif // LOAD_DATA_H
