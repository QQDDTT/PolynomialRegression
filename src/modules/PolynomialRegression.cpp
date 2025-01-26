#include "PolynomialRegression.h"
#include <stdexcept>

// 构造函数
// 参数：
//   degree - 多项式的最高阶数
//   numFeatures - 输入特征的维度
// 初始化多项式回归模型，分配参数向量并初始化为零
PolynomialRegression::PolynomialRegression(int degree, int numFeatures)
    : degree(degree), numFeatures(numFeatures) {
    if (degree < 1 || numFeatures < 1) {
        throw std::invalid_argument("阶数和特征数量必须为正整数。"); // 如果阶数或特征数量无效，则抛出异常
    }
    coefficients = arma::vec((degree + 1) * numFeatures, arma::fill::zeros); // 初始化回归系数为零
}

// 训练函数
// 参数：
//   X - 特征矩阵（每行是一个样本，每列是一个特征）
//   y - 标签向量（每个元素是对应样本的标签）
// 功能：使用最小二乘法计算多项式回归模型的系数
void PolynomialRegression::train(const arma::mat& X, const arma::vec& y) {
    arma::mat XExpanded = expandFeatures(X); // 扩展特征矩阵，加入多项式特征
    coefficients = arma::solve(XExpanded, y); // 使用最小二乘法求解系数
}

// 预测函数
// 参数：
//   X - 特征矩阵（每行是一个样本，每列是一个特征）
// 返回值：
//   预测结果向量（每个元素是对应样本的预测值）
// 功能：根据输入的特征矩阵预测输出
arma::vec PolynomialRegression::predict(const arma::mat& X) const {
    arma::mat XExpanded = expandFeatures(X); // 扩展特征矩阵，加入多项式特征
    return XExpanded * coefficients; // 根据回归系数计算预测值
}

// 设置回归系数
// 参数：
//   coefficients - 新的回归系数向量
// 功能：设置多项式回归模型的回归系数
void PolynomialRegression::setCoefficients(const arma::vec& coefficients) {
    if (coefficients.n_elem != this->coefficients.n_elem) {
        throw std::invalid_argument("回归系数的大小与预期值不匹配。"); // 如果新的系数大小不匹配，抛出异常
    }
    this->coefficients = coefficients; // 更新回归系数
}

// 获取回归系数
// 返回值：
//   当前的回归系数向量
// 功能：返回当前的回归系数
arma::vec PolynomialRegression::getCoefficients() const {
    return coefficients; // 返回回归系数
}

// 获取多项式的函数表达式
// 参数：
//   featureNames - 特征名称的字符串向量
// 返回值：
//   多项式的函数表达式字符串
// 功能：根据特征名称生成多项式回归模型的表达式
std::string PolynomialRegression::getPolynomialExpression(const std::vector<std::string>& featureNames) const {
    std::string expression = "f(x) = "; // 多项式表达式初始化
    int index = 0; // 用于索引回归系数

    // 生成多项式的每一项
    for (int d = 0; d <= degree; ++d) { // 遍历每个多项式的阶数
        for (int i = 0; i < featureNames.size(); ++i) {
            double coef = coefficients(index); // 获取当前的回归系数
            if (coef != 0) { // 如果系数不为零，才显示
                if (coef > 0 && index > 0) {
                    expression += " + "; // 如果系数为正且不是第一项，则加上加号
                } else if (coef < 0) {
                    expression += " - "; // 如果系数为负，则加上减号
                }

                // 显示系数
                if (std::abs(coef) != 1 || d == 0) { // 系数不为1时显示系数，或者是常数项
                    expression += std::to_string(std::abs(coef));
                }

                // 显示特征和阶数
                if (d > 0) {
                    expression += featureNames[i] + "^" + std::to_string(d);
                } else {
                    expression += featureNames[i]; // 常数项
                }
            }
            ++index; // 移动到下一个系数
        }
    }
    return expression; // 返回完整的多项式表达式
}


// 获取拟合函数与数据的拟合度
// 参数：
//   X - 特征矩阵
//   y - 标签向量
// 返回值：
//   拟合度（R-squared）
double PolynomialRegression::getRSquared(const arma::mat& X, const arma::vec& y) const {
    arma::mat XExpanded = expandFeatures(X); // 扩展特征矩阵
    arma::vec yPredicted = XExpanded * coefficients; // 计算预测值
    double yMean = arma::mean(y); // 计算标签的平均值
    double ssTotal = arma::sum(arma::square(y - yMean)); // 计算总平方和
    double ssResidual = arma::sum(arma::square(y - yPredicted)); // 计算残差平方和
    return 1.0 - ssResidual / ssTotal; // 计算并返回拟合度（R-squared）
}

// 获取拟合函数与数据的标准误差
// 参数：
//   X - 特征矩阵
//   y - 标签向量
// 返回值：
//   标准误差
double PolynomialRegression::getStandardError(const arma::mat& X, const arma::vec& y) const {
    arma::mat XExpanded = expandFeatures(X); // 扩展特征矩阵
    arma::vec yPredicted = XExpanded * coefficients; // 计算预测值
    double ssResidual = arma::sum(arma::square(y - yPredicted)); // 计算残差平方和
    double n = X.n_rows; // 获取样本数量
    return std::sqrt(ssResidual / (n - 1)); // 计算并返回标准误差
}

// 特征扩展函数
// 参数：
//   X - 输入特征矩阵（每行是一个样本，每列是一个特征）
// 返回值：
//   扩展后的特征矩阵（每个特征扩展为多项式形式）
// 功能：将原始特征映射到多项式特征空间
arma::mat PolynomialRegression::expandFeatures(const arma::mat& X) const {
    arma::mat XExpanded(X.n_rows, (degree + 1) * numFeatures, arma::fill::ones); // 初始化扩展后的特征矩阵，第一列是全为1的常数项
    for (int d = 1; d <= degree; ++d) {
        // 对每个特征进行多项式扩展
        XExpanded.cols(d * numFeatures, (d + 1) * numFeatures - 1) = arma::pow(X, d); // 使用pow函数进行特征扩展
    }
    return XExpanded; // 返回扩展后的特征矩阵
}
