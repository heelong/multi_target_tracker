#include "tracker.h"
#include <iomanip>

using namespace ATracker;

Tracker::Tracker(const KalmanParam& _param)
{
	param = _param;
	rng = cv::RNG(12345);
	trackIds = 1;
}
/*
 * 考虑到2D的结果没有深度信息，重叠部分较多，轨迹分两种：1、组合轨迹；2、单轨迹
 * 分两步分进行，首先处理组合轨迹内的检测结果，然后处理单轨迹的结果
 */
void Tracker::track(Detections& _detections, const int& w, const int& h, cv::Mat& img)
{
	evolveTracks();
	groups.size();
	const Detections& detections = manage_groups(_detections, img);

	cv::Mat assigments = associate_tracks(detections);//进行匹配

	if (assigments.total() != 0)//轨迹数不为0
		Hyphothesis::instance()->new_hyphothesis(assigments, single_tracks, detections, w, h,
		param.newhypdummycost(), prev_unassigned, param);
	assigments.setTo(0, (assigments == 255));

	if (single_tracks.size() == 0)
	{
		for (const auto& t : detections)
		{
			single_tracks.push_back(Track_ptr(new Track(t.x(), t.y(), t.w(), t.h(), param)));
		}
		time = cv::getTickCount();
	}
	else
	{
		update_tracks(assigments, detections);
		delete_tracks();
		detect_occlusions(img);
	}
}

//处理组合轨迹内的检测，并返回非组合轨迹
Detections Tracker::manage_groups(const Detections& _detections, cv::Mat& img)
{
	if (groups.size() > 0)
	{
		//CHECK TRACKS AND THE GROUPS检查单条轨迹与组合轨迹之间是否有重合，如果重合，则将单轨迹插入到组合轨迹中，并删除单轨迹
		for (int i = single_tracks.size() - 1; i >= 0; --i)
		{
			if (!single_tracks.at(i)->isgood) continue;
			const cv::Rect& det = single_tracks.at(i)->getRect();
			for (const auto& group : groups)
			{
				if (overlapRoi(det, group->getRect(), .4))//面积重叠40%以上
				{
					group->insert(single_tracks.at(i), img);
					single_tracks.erase(single_tracks.begin() + i);
					break;
				}
			}
		}
		//检查组合轨迹之间的重合
		//CHECK OVERLAPPING BETWEEN GROUPS
		for (int i = groups.size() - 1; i > 0; --i)
		{
			for (int j = i - 1; j >= 0; --j)
			{
				if (overlapRoi(groups.at(j)->getRect(), groups.at(i)->getRect(), .5))
				{
					groups.at(j)->merge(groups.at(i), img);
					groups.erase(groups.begin() + i);
					break;
				}
			}
		}

		Detections detections;
		//内联函数，判断检测结果是否在组合轨迹中
		auto isInside = [](const cv::Rect& group, const cv::Rect& det)
		{
			int x_tl = fmax(group.x, det.x);
			int y_tl = fmax(group.y, det.y);
			int x_br = fmin(group.x + group.width, det.x + det.width);
			int y_br = fmin(group.y + group.height, det.y + det.height);
			return (x_tl < x_br && y_tl < y_br);
		};

		std::vector<bool> checked(_detections.size(), false);

		//CHECK IF THE DETECTION IS INSIDE THE GROUP
		//检查检测结果是否在组合轨迹中
		for (const auto& group : groups)
		{
			const cv::Rect& r = group->getRect();
			UIntVec det_indices;
			uint i = 0;

			for (auto d : _detections)
			{
				const cv::Rect& det = d.getRect();
				if (isInside(r, det))
				{
					det_indices.push_back(i);
					checked.at(i) = true;
				}
				++i;
			}
			if (det_indices.size() > 0)
			{
				group->analyze_associations(_detections, det_indices, single_tracks, param, width, height, img);//对组合轨迹内的检测结果进行配对
			}
		}

		//CHECK GROUPS
		//检查组合轨迹，当组合轨迹中没有轨迹时，删除组合轨迹，或当组合轨迹中只有一条轨迹时，将其并入单轨迹中
		for (int i = groups.size() - 1; i >= 0; --i)
		{
			const auto& group = groups.at(i);

			if (group->size() == 0)
			{
				groups.erase(groups.begin() + i);
			}
			else if (group->size() == 1)
			{
				single_tracks.push_back(group->getTracks().at(0));
				groups.erase(groups.begin() + i);
			}
		}

		//RETURN ONLY NOT ASSOCIATED DETECTIONS
		//只返回不在组合轨迹中的检测结果
		uint i = 0;
		for (const auto& check : checked)
		{
			if (!check)
			{
				detections.push_back(_detections.at(i));
			}
			++i;
		}

		//UPDATE FILTER
		for (const auto& group : groups)
		{
			group->correct();
			group->resetDetection();
		}

		return detections;
	}

	return _detections;
}


void Tracker::delete_tracks()
{
	for (int i = single_tracks.size() - 1; i >= 0; --i)
	{
		const cv::Point2f& p = single_tracks.at(i)->getPointPrediction();
		const uint& ntime_missed = single_tracks.at(i)->ntime_missed;
		if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height || ntime_missed >= param.maxmissed())
		{
			single_tracks.erase(single_tracks.begin() + i);
		}
	}
}

/*
 * 关联检测结果与单轨迹
 */
cv::Mat Tracker::associate_tracks(const Detections& _detections)
{
	if (_detections.size() == 0) return cv::Mat();
	cv::Mat assigmentsBin(cv::Size(_detections.size(), single_tracks.size()), CV_8UC1, cv::Scalar(0));
	assignments_t assignments;
	distMatrix_t cost(_detections.size() * single_tracks.size());
	cv::Mat costs(cv::Size(_detections.size(), single_tracks.size()), CV_32FC1);
	cv::Mat mu;
	cv::Mat sigma;  //先验误差估计协方差矩阵
	cv::Point2f t;

	//COMPUTE COSTS
	const uint& tSize = single_tracks.size();
	const uint& dSize = _detections.size();
	for (uint i = 0; i < tSize; ++i)//列
	{
		mu = single_tracks.at(i)->getPrediction();
		sigma = single_tracks.at(i)->S();
		t = cv::Point2f(mu.at<float>(0), mu.at<float>(1));

		for (uint j = 0; j < dSize; ++j)//行
		{
			cv::Mat detection(cv::Size(1, 2), CV_32FC1);
			detection.at<float>(0) = _detections.at(j).x();
			detection.at<float>(1) = _detections.at(j).y();
			costs.at<float>(i, j) = cv::Mahalanobis(detection, cv::Mat(t), sigma.inv());//计算马氏距离
			cost.at(i + j * single_tracks.size()) = costs.at<float>(i, j);
		}
	}
	//for (int i = 0; i < tSize; i++)
	//{
	//	for (int j = 0; j < dSize; j++)
	//		std::cout << std::setw(10) << cost[i + j*tSize] << "   ";
	//	std::cout << std::setw(10) << std::endl;
	//}

	AssignmentProblemSolver APS;
	APS.Solve(cost, single_tracks.size(), _detections.size(), assignments, AssignmentProblemSolver::optimal);

	for (uint i = 0; i < assignments.size(); ++i)
	{
		if (assignments[i] != -1 && costs.at<float>(i, assignments[i]) < 40)
			assigmentsBin.at<uchar>(i, assignments[i]) = 1;
	}

	check_multiple_detections(assigmentsBin, costs);

	return assigmentsBin;
}

void Tracker::check_multiple_detections(cv::Mat& assigments, const cv::Mat& costs)
{
	cv::Mat rowCosts;
	cv::Mat l_same_detections;

	const uint& aRows = assigments.rows;

	for (uint i = 0; i < aRows; ++i)
	{
		const cv::Mat& row = assigments.row(i);
		cv::Mat assignment;
		cv::findNonZero(row, assignment);
		if (assignment.total() == 1)
		{
			rowCosts = costs.row(i);
			rowCosts.at<float>(assignment.at<cv::Point>(0).x) = FLT_MAX;
			const cv::Mat& same_detections = rowCosts < 50;
			cv::findNonZero(same_detections, l_same_detections);
			for (uint j = 0; j < l_same_detections.total(); ++j)
			{
				assigments.at<uchar>(i, l_same_detections.at<cv::Point>(j).x) = 255;
			}
		}

	}
}

//DETECT OCCLUSIONS AND MANAGE GROUPS
void Tracker::detect_occlusions(const cv::Mat& img)
{
	const uint& tSize = single_tracks.size();
	const uint& tSizeL = tSize - 1;

	UIntMat occlusions;
	for (uint i = 0; i < tSizeL; ++i)
	{
		if (!single_tracks.at(i)->isgood) continue;
		const cv::Rect& r1 = single_tracks.at(i)->getRect();
		UIntVec occls;
		occls.push_back(i);
		for (uint j = i + 1; j < tSize; ++j)
		{
			if (!single_tracks.at(j)->isgood) continue;
			const cv::Rect& r2 = single_tracks.at(j)->getRect();
			if (overlapRoi(r1, r2, .4))
			{
				occls.push_back(j);
			}
		}

		if (occls.size() > 1)
		{
			occlusions.push_back(occls);
		}
	}

	auto isIntersect = [](UIntVec& _v1, UIntVec& _v2)
	{
		UIntVec intersect;
		std::sort(_v1.begin(), _v1.end());
		std::sort(_v2.begin(), _v2.end());
		std::set_intersection(_v1.begin(), _v1.end(), _v2.begin(), _v2.end(),
			std::back_inserter(intersect));
		return intersect.size() > 0;
	};

	//CHECK INTERSECTIONS BETWEEN OCCLUSIONS
	for (int i = occlusions.size() - 1; i > 0; --i)
	{
		for (int j = i - 1; j >= 0; --j)
		{
			if (isIntersect(occlusions.at(i), occlusions.at(j)))
			{
				UIntVec newVec;
				std::merge(occlusions.at(i).begin(), occlusions.at(i).end(),
					occlusions.at(j).begin(), occlusions.at(j).end(),
					std::back_inserter(newVec));

				UIntVec::iterator pte = std::unique(newVec.begin(), newVec.end());
				newVec.erase(pte, newVec.end());
				occlusions.at(j) = newVec;
				occlusions.erase(occlusions.begin() + i);
				break;
			}
		}
	}


	//CREATE NEW GROUPS
	BoolVec tomove(single_tracks.size(), false);
	for (const auto& occlusion : occlusions)
	{
		const UIntVec& occl = occlusion;
		Tracks tomerge;
		for (const auto& o : occl)
		{
			tomove.at(o) = true;
			tomerge.push_back(single_tracks.at(o));
		}
		groups.push_back(Group_ptr(new GroupTrack(tomerge, img, param.dt())));
		groups.at(groups.size() - 1)->setColor(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
			rng.uniform(0, 255)));
	}

	//DELETE TRACKS FROM tracks
	for (int i = tomove.size() - 1; i >= 0; --i)
	{
		if (tomove.at(i))
		{
			single_tracks.erase(single_tracks.begin() + i);
		}
	}
}


bool Tracker::overlapRoi(const cv::Rect &_r1, const cv::Rect &_r2, const float& percentage)
{
	int x_tl = fmax(_r1.x, _r2.x);
	int y_tl = fmax(_r1.y, _r2.y);
	int x_br = fmin(_r1.x + _r1.width, _r2.x + _r2.width);
	int y_br = fmin(_r1.y + _r1.height, _r2.y + _r2.height);
	if (x_tl < x_br && y_tl < y_br)
	{
		const float& area = (x_br - x_tl) * (y_br - y_tl);
		return ((area / (float)_r1.area()) > percentage ||
			(area / (float)_r2.area()) > percentage);
	}
	return false;
}


void Tracker::update_tracks(const cv::Mat& assigments, const Detections& _detections)
{
	float prev_time = time;
	time = cv::getTickCount();
	const float& dt = (time - prev_time) / cv::getTickFrequency();
	const uint& aRows = assigments.rows;
	const uint& aCols = assigments.cols;

	for (uint i = 0; i < aRows; ++i)
	{
		for (uint j = 0; j < aCols; ++j)
		{
			if (assigments.at<uchar>(i, j) == uchar(1))//对单轨迹进行跟新，
			{
				single_tracks.at(i)->correct(_detections.at(j).x(), _detections.at(j).y(), _detections.at(j).w(), _detections.at(j).h());
				single_tracks.at(i)->ntime_missed = 0;//有对象匹配上了，其缺失值设为0
				single_tracks.at(i)->setDt(dt);

				if (single_tracks.at(i)->nTimePropagation() >= param.minpropagate() && !single_tracks.at(i)->isgood)//轨迹还没有初始化且持续跟踪次数大于设定次数，对轨迹进行确认
				{
					single_tracks.at(i)->setLabel(trackIds++);

					single_tracks.at(i)->setColor(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
						rng.uniform(0, 255)));
					single_tracks.at(i)->isgood = true;
				}
			}
		}
	}

	cv::Mat ass_sum;
	cv::reduce(assigments, ass_sum, 1, CV_REDUCE_SUM, CV_32S);

	for (uint i = 0; i < ass_sum.total(); ++i)
	{
		if (ass_sum.at<int>(i) == 0)
		{
			single_tracks.at(i)->ntime_missed++;//
		}
	}
}

const Entities Tracker::getTracks()
{
	tracks.clear();
	tracks.insert(tracks.end(), single_tracks.begin(), single_tracks.end());
	tracks.insert(tracks.end(), groups.begin(), groups.end());
	return tracks;
}
