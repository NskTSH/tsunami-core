#pragma once
#include <string>

enum SourceType
{
	POINT,
	ELLIPSOIDAL,
	CYLINDRICAL,
	RECTANGULAR
};

class ISource
{
protected:
	SourceType source_type_;
	std::string name_;
public:
	ISource(const std::string& n, SourceType st)
		: source_type_(st), name_(n)
	{
	}

	// ========== ИСПРАВЛЕНИЕ: Виртуальный деструктор для поддержки dynamic_cast ==========
	virtual ~ISource() = default;

	SourceType GetSourceType() const
	{
		return source_type_;
	}
};
