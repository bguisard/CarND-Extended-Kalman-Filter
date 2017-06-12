#include "kalman_filter.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Kalman Filter update equations
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();    // storing Ht matrix to improve speed
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  // New Estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // EKF update equations

  // converts back from cartesian to polar coords
  // store sqrt(px^2 + py^2) to avoid repeated calcs
  float sqrtPx2Py2 = std::sqrt(x_[0] * x_[0] + x_[1] * x_[1]);

  // in this step we implement h(x') 
  VectorXd h_x_ = VectorXd(3);
  h_x_ << sqrtPx2Py2,
          std::atan2(x_[1], x_[0]),
          (x_[0] * x_[2] + x_[1] * x_[3]) / sqrtPx2Py2; // px*vx + py*vy / sqrt(px^2 + py^2)

  VectorXd y = z - h_x_;

  // make sure phi' is within [-pi, pi]
  if (std::abs(y(1) > M_PI)) {
    while (y(1) > M_PI) y(1) -= 2 * M_PI;
    while (y(1) < -M_PI) y(1) += 2 * M_PI;
  }

  MatrixXd Ht = H_.transpose();    // storing Ht matrix to improve speed
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  // New Estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
