#include <vector>
int dt_predict(const std::vector<int>& features) {
int decision = 0;
if (features[2] <= 5) {
decision = int(((features[0]  >>  3)) + ((features[1]  >>  3)) + ((features[2]  <<  1)) + ((features[3]  >>  1)) + (-1));
} else {
decision = int(((features[0]  >>  2)) + ((features[1]  >>  1)) + ((features[2]  <<  2)) + ((features[3]  >>  0)) + (-30));
}

return decision;
}
