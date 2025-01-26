#include "PolynomialRegression.h"
#include "LoadData.h"
#include <iostream>
#include <vector>

int main() {
    // 数据文件路径
    const std::string dataFilePath = std::string(PROJECT_ROOT) + "/data/AT1X001.csv";

    arma::mat X; // 特征矩阵
    arma::vec y; // 标签向量

    // 尝试加载数据
    try {
        loadData(dataFilePath, X, y);
    } catch (const std::exception& ex) {
        std::cerr << "加载数据时发生错误: " << ex.what() << std::endl;
        return 1; // 返回错误码 1
    }

    std::cout << "加载的数据维度 (行 x 列): " << X.n_rows << " x " << X.n_cols << std::endl;

    // 多项式回归的参数
    int degree = 2; // 多项式最高阶数
    int numFeatures = X.n_cols; // 特征数量

    PolynomialRegression model(degree, numFeatures); // 初始化模型

    // 尝试训练模型
    try {
        model.train(X, y);
    } catch (const std::exception& ex) {
        std::cerr << "训练模型时发生错误: " << ex.what() << std::endl;
        return 1; // 返回错误码 1
    }

    // 生成预测结果
    arma::vec predictions = model.predict(X);

    // 输出预测结果和回归系数
    std::cout << "预测结果: " << predictions.t() << std::endl;
    std::cout << "回归系数: " << model.getCoefficients().t() << std::endl;

    // 特征名称，假设特征名称为 "x1", "x2", ..., "xn"
    std::vector<std::string> featureNames;
    for (int i = 0; i < numFeatures; ++i) {
        featureNames.push_back("x" + std::to_string(i + 1));
    }

    // 输出多项式的函数表达式
    std::string polynomialExpression = model.getPolynomialExpression(featureNames);
    std::cout << "拟合函数: " << polynomialExpression << std::endl;

    // 计算拟合度 R^2
    double rSquared = model.getRSquared(X, y);
    std::cout << "拟合度 R^2: " << rSquared << std::endl;

    // 计算标准误差
    double standardError = model.getStandardError(X, y);
    std::cout << "标准误差: " << standardError << std::endl;

    return 0;
}
