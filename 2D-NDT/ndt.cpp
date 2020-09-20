#include <iostream>
#include <armadillo>
#include <map>
#include<unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include<vector>
#include<string>

#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include"windows.h"


//using namespace dlib;
using namespace std;
using namespace arma;

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html

// NOTE: the C++11 "auto" keyword is not recommended for use with Armadillo objects and functions

mat refScan;
mat curScan;
mat xgridcoords, ygridcoords;

unordered_map<string, mat> meanq;
unordered_map<string, mat> covar;
unordered_map<string, mat> covarInv;


void preNDT(mat laserScan, double cellSize, mat& xgridcoords, mat& ygridcoords)
{
	double xmin = laserScan.col(0).min();
	double ymin = laserScan.col(1).min();
	double xmax = laserScan.col(0).max();
	double ymax = laserScan.col(1).max();

	double halfCellSize = cellSize / 2;

	float lowerBoundX = floor(xmin / cellSize)*cellSize - cellSize;
	float upperBoundX = ceil(xmax / cellSize)*cellSize + cellSize;
	float lowerBoundY = floor(ymin / cellSize)*cellSize - cellSize;
	float upperBoundY = ceil(ymax / cellSize)*cellSize + cellSize;

	int K = (upperBoundX - lowerBoundX) / cellSize + 1;
	//xgridcoords = mat(4, K);
	//ygridcoords = mat(4, K);
	xgridcoords.zeros(4, K);
	ygridcoords.zeros(4, K);

	for (int i = 0; i < K; ++i)
	{
		xgridcoords(0, i) = lowerBoundX + cellSize*i;
		xgridcoords(1, i) = lowerBoundX + halfCellSize + cellSize*i;
		xgridcoords(2, i) = lowerBoundX + cellSize*i;
		xgridcoords(3, i) = lowerBoundX + halfCellSize + cellSize*i;

		ygridcoords(0, i) = lowerBoundY + cellSize*i;
		ygridcoords(1, i) = lowerBoundY + cellSize*i;
		ygridcoords(2, i) = lowerBoundY + halfCellSize + cellSize*i;
		ygridcoords(3, i) = lowerBoundY + halfCellSize + cellSize*i;
	}

}


void buildNDT(mat laserScan, double cellSize, mat& xgridcoords, mat& ygridcoords, unordered_map<string, mat>& meanq, unordered_map<string, mat>& covar, unordered_map<string, mat>& covarInv)
{
	//if (laserScan.is_empty())
	//{
	//	xgridcoords=mat(4, 0);
	//	ygridcoords=mat(4, 0);
	//	return;
	//}
	preNDT(laserScan, 1, xgridcoords, ygridcoords);

	int xNumCells = xgridcoords.size() / 4;
	int yNumCells = ygridcoords.size() / 4;

	//unordered_map<Eigen::Vector3i, mat> mean;
	//unordered_map<Eigen::Vector3i, mat> covar;
	//unordered_map<Eigen::Vector3i, mat> covarInv;
	mat premean(1, 2);
	premean << 0 << 0 << endr;
	mat precov(2, 2);
	precov << 0 << 0 << endr
		<< 0 << 0 << endr;
	for (int mode = 0; mode < 4; ++mode)
		for (int i = 0; i < xNumCells; ++i)
			for (int j = 0; j < yNumCells; ++j)
			{
				string prekey = to_string(mode) + " " + to_string(i) + " " + to_string(j);
				meanq.insert(make_pair(prekey, premean));
				covar.insert(make_pair(prekey, precov));
				covarInv.insert(make_pair(prekey, precov));

			}

	for (int cellShiftMode = 0; cellShiftMode < 4; ++cellShiftMode)
	{
		vec x = laserScan.col(0);
		vec xg = xgridcoords.row(cellShiftMode).t();
		vec y = laserScan.col(1);
		vec yg = ygridcoords.row(cellShiftMode).t();
		umat idx = histc(x, xg, 1);
		umat idy = histc(y, yg, 1);
		for (int i = 0; i < xNumCells; ++i)
		{
			umat xflags = idx.col(i);
			for (int j = 0; j < yNumCells; ++j)
			{
				umat yflags = idy.col(j);
				umat xyflags = xflags%yflags;
				uvec xyflags_id = find(xyflags == 1);
				mat xymemberInCell = laserScan.rows(xyflags_id);

				if (xymemberInCell.n_rows > 3)
				{
					mat xymean = mean(xymemberInCell);
					mat xyCov = cov(xymemberInCell, 1);

					mat U;
					vec s;
					mat V;

					svd(U, s, V, xyCov);

					mat S(2, 2);
					S << s(0) << 0 << endr
						<< 0 << s(1) << endr;


					if (S(1, 1) < 0.001*S(0, 0))
					{
						S(1, 1) = 0.001 * S(0, 0);
						xyCov = U*S*V.t();
					}
					mat R;
					if (!chol(R, xyCov))
					{
						continue;
					}
					string key = to_string(cellShiftMode) + " " + to_string(i) + " " + to_string(j);
					meanq[key] = xymean;
					covar[key] = xyCov;
					covarInv[key] = inv(xyCov);

				}
			}
		}
	}
}

//void objectiveNDT(mat laserScan, vec laserTrans, const mat& xgridcoords, const mat& ygridcoords,  unordered_map<string, mat>& meanq,  unordered_map<string, mat>& covar,  unordered_map<string, mat>& covarInv,double& score,vec& gradient,mat& hessian)
//{
//	double tx = laserTrans(0);
//	double ty = laserTrans(1);
//	double theta = laserTrans(2);
//	double sintheta = sin(theta);
//	double costheta = cos(theta);
//
//	mat tform(3, 3);
//	tform << costheta << -sintheta << tx << endr
//		<< sintheta << costheta << ty << endr
//		<< 0 << 0 << 1 << endr;
//
//	mat hom(laserScan.n_rows, 3);
//	hom.fill(1);
//	hom.cols(0, 1) = laserScan;
//
//	mat trPts=hom*tform;
//
//	mat laserTransformed = trPts.cols(0, 1);
//
//	hessian.zeros(3, 3);
//	gradient.zeros(3, 1);
//	score = 0;
//
//	for (int i = 0; i < laserTransformed.n_rows; ++i)
//	{
//		vec xprime;
//		xprime<< laserTransformed(i, 0);
//		vec yprime;
//		yprime << laserTransformed(i, 1);
//
//		double x = laserScan(i, 0);
//		double y = laserScan(i, 1);
//
//		mat jacobianT(2, 3);
//		jacobianT << 1 << 0 << -x*sintheta - y*costheta << endr
//			<< 0 << 1 << x*costheta - y*sintheta << endr;
//
//		mat qp(2, 1);
//		qp << -x*costheta + y*sintheta << endr
//			<< -x*sintheta - y*costheta << endr;
//
//		for (int cellShiftMode = 0; cellShiftMode < 4; ++cellShiftMode)
//		{
//			vec xg = xgridcoords.row(cellShiftMode).t();
//			vec yg = ygridcoords.row(cellShiftMode).t();
//			umat idx = histc(xprime, xg, 1);
//			umat idy = histc(yprime, yg, 1);
//			uvec m = find(idx == 1);
//			uvec n = find(idy == 1);
//			if (m.is_empty() || n.is_empty())
//			{
//				continue;
//			}
//			int x_m = m(0);
//			int y_n = n(0);
//			string key = to_string(cellShiftMode) + " " + to_string(x_m) + " " + to_string(y_n);
//			mat meanmn = meanq[key].t();
//			mat covarmn = covar[key];
//			mat covarmninv = covarInv[key];
//
//			umat any_m = any(meanmn);
//			umat any_c = any(covarmn);
//			vec flags;
//			flags << any_m(0) << any_c(0) << any_c(1);
//
//			if (!any(flags))
//			{
//				continue;
//			}
//
//
//			mat q(2, 1);
//			q << xprime(0) << endr
//				<< yprime(0) << endr;
//			q = q - meanmn;
//			mat a = q.t()*covarmninv*q / 2;
//			double gaussianValue = exp(-a(0));
//			score = score -gaussianValue;
//
//			for (int j = 0; j < 3; j++)
//			{
//				mat gradelta = q.t()*covarmninv*jacobianT.col(j)*gaussianValue;
//				gradient(j) = gradient(j) + gradelta(0);
//				mat qpj = jacobianT.col(j);
//				for (int k = j; k < 3; ++k)
//				{
//					mat qpk = jacobianT.col(k);
//					if (j == 2 && k == 2)
//					{
//						mat h1 = q.t()*covarmninv*qpj;
//						mat h2 = q.t()*covarmninv*qpk;
//						mat h3 = q.t()*covarmninv*qp;
//						mat h4 = qpk.t()*covarmninv*qpj;
//						hessian(j, k) = hessian(j, k) + gaussianValue*(-h1(0)*h2(0) + h3(0) + h4(0));
//					}
//					else
//					{
//						mat h1 = q.t()*covarmninv*qpj;
//						mat h2 = q.t()*covarmninv*qpk;
//						mat h4 = qpk.t()*covarmninv*qpj;
//						hessian(j, k) = hessian(j, k) + gaussianValue*(-h1(0)*h2(0) + h4(0));
//					}
//				}
//			}
//
//
//		}
//	}
//
//	for (int j = 0; j < 3; ++j)
//		for (int k = 0; k < j ; ++k)
//			hessian(j, k) = hessian(k, j);
//}

typedef dlib::matrix<double, 0, 1> column_vector;


double ndt(const column_vector& m)
{
	//cout << m << endl;
	vec laserTrans(3, 1);
	laserTrans << m(0) << endr << m(1) << endr << m(2) << endr;
	double tx = laserTrans(0);
	double ty = laserTrans(1);
	double theta = laserTrans(2);
	double sintheta = sin(theta);
	double costheta = cos(theta);

	mat tform(3, 3);
	tform << costheta << -sintheta << tx << endr
		<< sintheta << costheta << ty << endr
		<< 0 << 0 << 1 << endr;

	mat hom(curScan.n_rows, 3);
	hom.fill(1);
	hom.cols(0, 1) = curScan;

	mat trPts = hom*tform.t();

	mat laserTransformed = trPts.cols(0, 1);


	double score = 0;

	for (int i = 0; i < laserTransformed.n_rows; ++i)
	{
		vec xprime;
		xprime << laserTransformed(i, 0);
		vec yprime;
		yprime << laserTransformed(i, 1);

		double x = curScan(i, 0);
		double y = curScan(i, 1);

		mat jacobianT(2, 3);
		jacobianT << 1 << 0 << -x*sintheta - y*costheta << endr
			<< 0 << 1 << x*costheta - y*sintheta << endr;

		mat qp(2, 1);
		qp << -x*costheta + y*sintheta << endr
			<< -x*sintheta - y*costheta << endr;

		for (int cellShiftMode = 0; cellShiftMode < 4; ++cellShiftMode)
		{
			vec xg = xgridcoords.row(cellShiftMode).t();
			vec yg = ygridcoords.row(cellShiftMode).t();
			umat idx = histc(xprime, xg, 1);
			umat idy = histc(yprime, yg, 1);
			uvec m = find(idx == 1);
			uvec n = find(idy == 1);
			if (m.is_empty() || n.is_empty())
			{
				continue;
			}
			int x_m = m(0);
			int y_n = n(0);
			string key = to_string(cellShiftMode) + " " + to_string(x_m) + " " + to_string(y_n);
			mat meanmn = meanq[key].t();
			mat covarmn = covar[key];
			mat covarmninv = covarInv[key];

			umat any_m = any(meanmn);
			umat any_c = any(covarmn);
			vec flags;
			flags << any_m(0) << any_c(0) << any_c(1);

			if (!any(flags))
			{
				continue;
			}


			mat q(2, 1);
			q << xprime(0) << endr
				<< yprime(0) << endr;
			q = q - meanmn;
			mat a = q.t()*covarmninv*q / 2;
			double gaussianValue = exp(-a(0));
			score = score - gaussianValue;
		}
	}

	return score;
}

const column_vector ndt_derivative(const column_vector& m)
{
	column_vector res(3);
	res(0) = 0;
	res(1) = 0;
	res(2) = 0;
	//cout << m << endl;
	vec laserTrans(3, 1);
	laserTrans << m(0) << endr << m(1) << endr << m(2) << endr;
	double tx = laserTrans(0);
	double ty = laserTrans(1);
	double theta = laserTrans(2);
	double sintheta = sin(theta);
	double costheta = cos(theta);

	mat tform(3, 3);
	tform << costheta << -sintheta << tx << endr
		<< sintheta << costheta << ty << endr
		<< 0 << 0 << 1 << endr;

	mat hom(curScan.n_rows, 3);
	hom.fill(1);
	hom.cols(0, 1) = curScan;

	mat trPts = hom*tform.t();

	mat laserTransformed = trPts.cols(0, 1);



	for (int i = 0; i < laserTransformed.n_rows; ++i)
	{
		vec xprime;
		xprime << laserTransformed(i, 0);
		vec yprime;
		yprime << laserTransformed(i, 1);

		double x = curScan(i, 0);
		double y = curScan(i, 1);

		mat jacobianT(2, 3);
		jacobianT << 1 << 0 << -x*sintheta - y*costheta << endr
			<< 0 << 1 << x*costheta - y*sintheta << endr;

		mat qp(2, 1);
		qp << -x*costheta + y*sintheta << endr
			<< -x*sintheta - y*costheta << endr;

		for (int cellShiftMode = 0; cellShiftMode < 4; ++cellShiftMode)
		{
			vec xg = xgridcoords.row(cellShiftMode).t();
			vec yg = ygridcoords.row(cellShiftMode).t();
			umat idx = histc(xprime, xg, 1);
			umat idy = histc(yprime, yg, 1);
			uvec m = find(idx == 1);
			uvec n = find(idy == 1);
			if (m.is_empty() || n.is_empty())
			{
				continue;
			}
			int x_m = m(0);
			int y_n = n(0);
			string key = to_string(cellShiftMode) + " " + to_string(x_m) + " " + to_string(y_n);
			mat meanmn = meanq[key].t();
			mat covarmn = covar[key];
			mat covarmninv = covarInv[key];

			umat any_m = any(meanmn);
			umat any_c = any(covarmn);
			vec flags;
			flags << any_m(0) << any_c(0) << any_c(1);

			if (!any(flags))
			{
				continue;
			}


			mat q(2, 1);
			q << xprime(0) << endr
				<< yprime(0) << endr;
			q = q - meanmn;
			mat a = q.t()*covarmninv*q / 2;
			double gaussianValue = exp(-a(0));

			for (int j = 0; j < 3; j++)
			{
				mat gradelta = q.t()*covarmninv*jacobianT.col(j)*gaussianValue;
				res(j) = res(j) + gradelta(0);
			}
		}
	}



	// now compute the gradient vector


	return res;
}

dlib::matrix<double> ndt_hessian(const column_vector& m)
{
	dlib::matrix<double> res(3, 3);
	//res(0, 0) = 0; // second derivative with respect to x
	//res(1, 1) = 0;
	//res(2, 2) = 0;
	//res(1, 0) = res(0, 1) =0;
	//res(2, 0) = res(0, 2) =0;
	//res(2, 1) = res(1, 2) = 0;

	//cout << m << endl;

	double tx = m(0);
	double ty = m(1);
	double theta = m(2);
	double sintheta = sin(theta);
	double costheta = cos(theta);

	mat tform(3, 3);
	tform << costheta << -sintheta << tx << endr
		<< sintheta << costheta << ty << endr
		<< 0 << 0 << 1 << endr;

	mat hom(curScan.n_rows, 3);
	hom.fill(1);
	hom.cols(0, 1) = curScan;

	mat trPts = hom*tform.t();

	mat laserTransformed = trPts.cols(0, 1);



	for (int i = 0; i < laserTransformed.n_rows; ++i)
	{
		vec xprime;
		xprime << laserTransformed(i, 0);
		vec yprime;
		yprime << laserTransformed(i, 1);

		double x = curScan(i, 0);
		double y = curScan(i, 1);

		mat jacobianT(2, 3);
		jacobianT << 1 << 0 << -x*sintheta - y*costheta << endr
			<< 0 << 1 << x*costheta - y*sintheta << endr;

		mat qp(2, 1);
		qp << -x*costheta + y*sintheta << endr
			<< -x*sintheta - y*costheta << endr;

		for (int cellShiftMode = 0; cellShiftMode < 4; ++cellShiftMode)
		{
			vec xg = xgridcoords.row(cellShiftMode).t();
			vec yg = ygridcoords.row(cellShiftMode).t();
			umat idx = histc(xprime, xg, 1);
			umat idy = histc(yprime, yg, 1);
			uvec m = find(idx == 1);
			uvec n = find(idy == 1);
			if (m.is_empty() || n.is_empty())
			{
				continue;
			}
			int x_m = m(0);
			int y_n = n(0);
			string key = to_string(cellShiftMode) + " " + to_string(x_m) + " " + to_string(y_n);
			mat meanmn = meanq[key].t();
			mat covarmn = covar[key];
			mat covarmninv = covarInv[key];

			umat any_m = any(meanmn);
			umat any_c = any(covarmn);
			vec flags;
			flags << any_m(0) << any_c(0) << any_c(1);

			if (!any(flags))
			{
				continue;
			}


			mat q(2, 1);
			q << xprime(0) << endr
				<< yprime(0) << endr;
			q = q - meanmn;
			mat a = q.t()*covarmninv*q / 2;
			double gaussianValue = exp(-a(0));

			for (int j = 0; j < 3; j++)
			{
				mat qpj = jacobianT.col(j);
				for (int k = j; k < 3; ++k)
				{
					mat qpk = jacobianT.col(k);
					if (j == 2 && k == 2)
					{
						mat h1 = q.t()*covarmninv*qpj;
						mat h2 = q.t()*covarmninv*qpk;
						mat h3 = q.t()*covarmninv*qp;
						mat h4 = qpk.t()*covarmninv*qpj;
						res(j, k) = res(j, k) + gaussianValue*(-h1(0)*h2(0) + h3(0) + h4(0));
					}
					else
					{
						mat h1 = q.t()*covarmninv*qpj;
						mat h2 = q.t()*covarmninv*qpk;
						mat h4 = qpk.t()*covarmninv*qpj;
						res(j, k) = res(j, k) + gaussianValue*(-h1(0)*h2(0) + h4(0));
					}
				}
			}
		}
	}

	for (int j = 0; j < 3; ++j)
		for (int k = 0; k < j; ++k)
			res(j, k) = res(k, j);

	return res;
}


class ndt_model
{
public:
	typedef ::column_vector column_vector;
	typedef dlib::matrix<double> general_matrix;

	double operator() (
		const column_vector& x
		) const {
		return ndt(x);
	}

	void get_derivative_and_hessian(
		const column_vector& x,
		column_vector& der,
		general_matrix& hess
	) const
	{
		der = ndt_derivative(x);
		hess = ndt_hessian(x);
	}
};


column_vector  matchNDT(mat refScan, mat curScan, double cellSize, column_vector starting_point)
{
	buildNDT(refScan, 1, xgridcoords, ygridcoords, meanq, covar, covarInv);
	find_min_trust_region(dlib::objective_delta_stop_strategy(1e-6),
		ndt_model(),
		starting_point,
		10
	);
	return starting_point;
}

column_vector trans2absolute(column_vector baseTransform, column_vector relativeTransform)
{
	Eigen::Affine2f basetrans(Eigen::Translation2f(baseTransform(0), baseTransform(1))*Eigen::Rotation2Df(baseTransform(2)));
	Eigen::Affine2f relativetrans(Eigen::Translation2f(relativeTransform(0), relativeTransform(1))*Eigen::Rotation2Df(relativeTransform(2)));
	Eigen::Matrix3f curpose = basetrans.matrix()*relativetrans.matrix();

	Eigen::Matrix2f rotation = curpose.block(0, 0, 2, 2);

	double angle = atan2(rotation.coeff(1, 0), rotation.coeff(0, 0));


	column_vector absolutePose(3);
	absolutePose(0) = curpose(6);
	absolutePose(1) = curpose(7);
	absolutePose(2) = angle;
	return absolutePose;
}


int main(int argc, char** argv)
{
	refScan.load("./data/refScan.txt");
	curScan.load("./data/curScan.txt");
	column_vector starting_point = { 0,0,0 };

	long start_time = GetTickCount();
	column_vector pose = matchNDT(refScan, curScan, 1, starting_point);
	long end_time = GetTickCount();
	cout << pose << endl;
	cout << "程序段运行时间：" << (end_time - start_time) << "ms!" << endl;

	return 0;
}

