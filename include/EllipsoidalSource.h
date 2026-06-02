#pragma once
#include "ISource.h"

class EllipsoidalSource : public ISource
{
private:
	double radius1_;
	double radius2_;
	double center_x_;
	double center_y_;
	double angle_;
public:

	EllipsoidalSource(const std::string& n, double x, double y,
		double a1, double a2, double a)
		: ISource(n, ELLIPSOIDAL),     // <-- инициализируем базовую часть
		center_x_(x),
		center_y_(y),
		radius1_(a1),
		radius2_(a2),
		angle_(a){
	}

	double GetRadius1() const
	{
		return radius1_;
	}

	double GetRadius2() const
	{
		return radius2_;
	}

	double GetCenterX() const {
		return center_x_;
	}

	double GetCenterY() const {
		return center_y_;
	}

	double GetAngle() const {
		return angle_;
	}
};
