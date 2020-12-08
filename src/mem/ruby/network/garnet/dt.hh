#include <vector>
int dt_predict(const std::vector<int>& features) {
int decision = 0;
if (features[2] <= 5) {
decision = int(((features[0]  >>  4)) + ((features[1]  >>  3)) + ((features[2]  <<  1)) + ((features[3]  >>  1)) + (4));
} else {
decision = int(((features[0]  >>  4)) + ((features[1]  >>  1)) + ((features[2]  <<  2)) + ((features[3]  <<  1)) + (-26));
}

return decision;
}
