#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>

struct Inputs
{
    double C_i = 0.0;
    double C_c = 0.0;
    std::vector<double> p;
};

static void usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " input_file [output_tsv]\n";
}

static bool read_input_file(const std::string &path, Inputs &out)
{
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    bool have_header = false;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;
        if (!have_header)
        {
            char *end1 = nullptr;
            char *end2 = nullptr;
            const double c_i = std::strtod(line.c_str(), &end1);
            if (end1 == line.c_str())
                continue;
            const double c_c = std::strtod(end1, &end2);
            if (end2 == end1)
                continue;
            out.C_i = c_i;
            out.C_c = c_c;
            have_header = true;
            continue;
        }
        char *endn = nullptr;
        char *endp = nullptr;
        const long n = std::strtol(line.c_str(), &endn, 10);
        if (endn == line.c_str())
            continue;
        const double p = std::strtod(endn, &endp);
        if (endp == endn)
            continue;
        if (n < 0)
            continue;
        if (static_cast<size_t>(n) >= out.p.size())
        {
            out.p.resize(static_cast<size_t>(n) + 1, 0.0);
        }
        out.p[static_cast<size_t>(n)] = p;
    }
    return have_header;
}

static double compute_objective(int W, int I, const Inputs &in)
{
    double acc = 0.0;
    for (size_t n = 0; n < in.p.size(); ++n)
    {
        if (in.p[n] == 0.0)
            continue;

        // 収束が W 以前でも、最初のチェックは W+I なので k>=1
        int k = 1;
        if (static_cast<int>(n) > W)
        {
            int diff = static_cast<int>(n) - W;
            k = (diff + I - 1) / I; // ceil((n-W)/I)
        }

        // 実際に停止する反復回数
        int l = W + I * k;

        acc += in.p[n] * (in.C_i * l + in.C_c * k);
    }
    return acc;
}

static std::string default_output_path(const std::string &input_path)
{
    size_t slash = input_path.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "" : input_path.substr(0, slash + 1);
    std::string file = (slash == std::string::npos) ? input_path : input_path.substr(slash + 1);
    const std::string suffix = "_costs.txt";
    if (file.size() >= suffix.size() &&
        file.compare(file.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        std::string base = file.substr(0, file.size() - suffix.size());
        return dir + base + "_objective.tsv";
    }
    size_t dot = file.find_last_of('.');
    std::string base = (dot == std::string::npos) ? file : file.substr(0, dot);
    return dir + base + "_objective.tsv";
}

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3)
    {
        usage(argv[0]);
        return 1;
    }

    Inputs in;
    if (!read_input_file(argv[1], in))
    {
        std::cerr << "Failed to read input file: " << argv[1] << "\n";
        return 1;
    }
    if (in.p.empty())
    {
        std::cerr << "p data is empty or invalid: " << argv[1] << "\n";
        return 1;
    }

    int n_up = -1;
    for (size_t n = 0; n < in.p.size(); ++n)
    {
        if (in.p[n] != 0.0)
        {
            n_up = static_cast<int>(n);
        }
    }
    if (n_up < 0)
    {
        std::cerr << "All p[n] are zero; no valid n_up.\n";
        return 1;
    }

    const int W_min = 0;
    const int W_max = n_up;

    struct Result
    {
        int W;
        int I;
        double obj;
    };
    std::vector<Result> top;
    top.reserve(10);

    for (int W = W_min; W <= W_max; ++W)
    {
        const int I_min = 1;
        const int I_max = n_up - W + 1; // smallest I where W + I > n_up
        for (int I = I_min; I <= I_max; ++I)
        {
            const double obj = compute_objective(W, I, in);
            top.push_back(Result{W, I, obj});
            std::sort(top.begin(), top.end(), [](const Result &a, const Result &b)
                      { return a.obj < b.obj; });
            if (top.size() > 10)
                top.pop_back();
        }
    }

    const std::string output_path = (argc == 3) ? argv[2] : default_output_path(argv[1]);
    std::ofstream out(output_path);
    if (!out)
    {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }

    out << "interval\twarmup\tobjective\n";
    out << std::setprecision(17);
    for (size_t i = 0; i < top.size(); ++i)
    {
        out << top[i].I << "\t" << top[i].W << "\t" << top[i].obj << "\n";
    }
    out.close();

    std::cout << "top10 (best to worst)\n";
    for (size_t i = 0; i < top.size(); ++i)
    {
        std::cout << (i + 1) << ": W=" << top[i].W
                  << " I=" << top[i].I
                  << " obj=" << top[i].obj << "\n";
    }
    std::cerr << "Wrote results to " << output_path << "\n";
    return 0;
}
