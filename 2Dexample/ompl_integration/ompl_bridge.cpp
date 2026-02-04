#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/base/SpaceInformation.h>
#include <vector>

namespace py = pybind11;
namespace ob = ompl::base;
namespace og = ompl::geometric;

// 通用规划函数：支持任意维度和限位，并支持 Python 碰撞检测回调
std::vector<std::vector<double>> plan_generic(
    int dim, 
    std::vector<double> low_bounds, 
    std::vector<double> high_bounds,
    std::vector<double> start_vec, 
    std::vector<double> goal_vec,
    py::function is_valid_py) 
{
    // 1. 定义配置空间
    auto space(std::make_shared<ob::RealVectorStateSpace>(dim));
    ob::RealVectorBounds bounds(dim);
    for(int i=0; i<dim; ++i) {
        bounds.setLow(i, low_bounds[i]);
        bounds.setHigh(i, high_bounds[i]);
    }
    space->setBounds(bounds);

    // 2. 空间信息及碰撞校验器 (对接 Python)
    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker([is_valid_py, dim](const ob::State* state) {
        const auto* data = state->as<ob::RealVectorStateSpace::StateType>()->values;
        std::vector<double> vec(dim);
        for(int i=0; i<dim; ++i) vec[i] = data[i];
        // 调用 Python 里的碰撞检测函数
        return is_valid_py(vec).cast<bool>();
    }); 
    si->setup();

    // 3. 起点终点转换
    ob::ScopedState<> start(space), goal(space);
    for(int i=0; i<dim; ++i) { 
        start[i] = start_vec[i]; 
        goal[i] = goal_vec[i]; 
    }

    // 4. 定义问题
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(start, goal);

    // 5. 实例化规划器 (使用原生 RRT 并极大限制步长以获得极致 Zigzag 效果)
    auto planner(std::make_shared<og::RRT>(si));
    planner->setRange(0.01); // 步长减小到 0.01，强制产生极多细小折点
    planner->setProblemDefinition(pdef);
    planner->setup();

    // 6. 执行规划
    std::vector<std::vector<double>> result_path;
    // Franka 7-DOF 空间较大，给予 5.0s 时间确保能搜到路径
    if (planner->ob::Planner::solve(5.0)) {
        og::PathGeometric &path = static_cast<og::PathGeometric&>(*pdef->getSolutionPath());
        // path.interpolate(); // Disable OMPL interpolation to show raw sampled nodes
        for (size_t i = 0; i < path.getStateCount(); ++i) {
            std::vector<double> state_vec(dim);
            const auto* data = path.getState(i)->as<ob::RealVectorStateSpace::StateType>()->values;
            for(int j=0; j<dim; ++j) state_vec[j] = data[j];
            result_path.push_back(state_vec);
        }
    }
    return result_path;
}

PYBIND11_MODULE(my_ompl_core, m) {
    m.def("plan", &plan_generic, "A high-performance generic OMPL planner for Python with collision callback",
          py::arg("dim"), py::arg("low_bounds"), py::arg("high_bounds"), 
          py::arg("start_vec"), py::arg("goal_vec"), py::arg("is_valid_py"));
}

