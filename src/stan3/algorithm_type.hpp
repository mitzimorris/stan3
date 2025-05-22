#ifndef STAN3_ALGORITHM_TYPE_HPP
#define STAN3_ALGORITHM_TYPE_HPP

namespace stan3 {

enum class algorithm_t {
    STAN2_HMC = 1,
    MLE = 2,
    PATHFINDER = 3,
    ADVI = 4,
    STANDALONE_GQ = 5
};

}  // namespace stan3
#endif
