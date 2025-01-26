#ifndef POLYNOMIAL_REGRESSION_H
#define POLYNOMIAL_REGRESSION_H

#include <armadillo>
#include <vector>
#include <string>

// 多项式回归类
// 提供多项式回归模型的训练、预测和系数管理功能
class PolynomialRegression {
public:
    // 构造函数
    // 参数：
    //   degree - 多项式的最高阶数
    //   numFeatures - 输入特征的维度
    PolynomialRegression(int degree, int numFeatures);

    // 训练函数
    // 参数：
    //   X - 特征矩阵，大小为 (样本数量, 特征数量)
    //   y - 标签向量，大小为 (样本数量)
    void train(const arma::mat& X, const arma::vec& y);

    // 预测函数
    // 参数：
    //   X - 特征矩阵，大小为 (样本数量, 特征数量)
    // 返回值：
    //   预测结果向量，大小为 (样本数量)
    arma::vec predict(const arma::mat& X) const;

    // 设置回归系数
    // 参数：
    //   coefficients - 回归系数向量
    void setCoefficients(const arma::vec& coefficients);

    // 获取回归系数
    // 返回值：
    //   回归系数向量
    arma::vec getCoefficients() const;

    // 获取多项式的函数表达式
    // 参数：
    //   featureNames - 特征名称向量
    // 返回值：
    //   多项式的函数表达式字符串
    std::string getPolynomialExpression(const std::vector<std::string>& featureNames) const;

    // 获取拟合函数与数据的拟合度
    // 参数：
    //   X - 特征矩阵，大小为 (样本数量, 特征数量)
    //   y - 标签向量，大小为 (样本数量)
    // 返回值：
    //   拟合度
    double getRSquared(const arma::mat& X, const arma::vec& y) const;

    // 获取拟合函数与数据的标准误差
    // 参数：
    //   X - 特征矩阵，大小为 (样本数量, 特征数量)
    //   y - 标签向量，大小为 (样本数量)
    // 返回值：
    //   标准误
    double getStandardError(const arma::mat& X, const arma::vec& y) const;

private:
    int degree;               // 多项式的最高阶数
    int numFeatures;          // 输入特征的维度
    arma::vec coefficients;   // 回归系数向量

    // 特征扩展函数
    // 将输入特征矩阵扩展为多项式特征
    // 参数：
    //   X - 输入特征矩阵
    // 返回值：
    //   扩展后的特征矩阵
    arma::mat expandFeatures(const arma::mat& X) const;
};

#endif // POLYNOMIAL_REGRESSION_H
