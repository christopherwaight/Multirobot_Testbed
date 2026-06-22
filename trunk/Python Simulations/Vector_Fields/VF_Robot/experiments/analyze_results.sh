#!/bin/bash

# Comprehensive sim2real analysis script (no Python dependencies required)

CSV_FILE="sim2real_center_tracking_results/sim2real_center_tracking_results.csv"

echo "================================================================================"
echo "SIM-TO-REAL COMPARISON ANALYSIS"
echo "================================================================================"
echo ""

# Count configurations
TOTAL=$(tail -n +2 "$CSV_FILE" | wc -l | tr -d ' ')
COUNT_0XX=$(tail -n +2 "$CSV_FILE" | awk -F',' '$3=="0XX"' | wc -l | tr -d ' ')
COUNT_1XX=$(tail -n +2 "$CSV_FILE" | awk -F',' '$3=="1XX"' | wc -l | tr -d ' ')
COUNT_2XX=$(tail -n +2 "$CSV_FILE" | awk -F',' '$3=="2XX"' | wc -l | tr -d ' ')

echo "Total configurations analyzed: $TOTAL"
echo "  0XX (3-robot):          $COUNT_0XX"
echo "  1XX (4-robot-square):   $COUNT_1XX"
echo "  2XX (4-robot-advanced): $COUNT_2XX"
echo ""

echo "================================================================================"
echo "Q1: WHICH SIMULATION MATCHES REALITY BEST?"
echo "================================================================================"
echo ""

# Calculate absolute errors vs real robot
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    ana_err = abs($47)  # analytical_vs_real (column 47)
    nn_err = abs($48)   # nn_vs_real (column 48)
    rbf_err = abs($49)  # rbf_vs_real (column 49)

    ana_sum += ana_err
    nn_sum += nn_err
    rbf_sum += rbf_err

    ana_sq += ana_err * ana_err
    nn_sq += nn_err * nn_err
    rbf_sq += rbf_err * rbf_err

    count++
}
END {
    ana_mean = ana_sum / count
    nn_mean = nn_sum / count
    rbf_mean = rbf_sum / count

    ana_std = sqrt(ana_sq/count - ana_mean*ana_mean)
    nn_std = sqrt(nn_sq/count - nn_mean*nn_mean)
    rbf_std = sqrt(rbf_sq/count - rbf_mean*rbf_mean)

    printf "Average Absolute Radius Error vs Real Robot (all configs):\n"
    printf "  Analytical:  %.4f ± %.4f m\n", ana_mean, ana_std
    printf "  Neural Net:  %.4f ± %.4f m\n", nn_mean, nn_std
    printf "  RBF:         %.4f ± %.4f m\n", rbf_mean, rbf_std
    printf "\n"

    # Determine winner
    if (nn_mean < ana_mean && nn_mean < rbf_mean) {
        printf "WINNER (Overall): Neural Net with %.4fm average error\n", nn_mean
    } else if (rbf_mean < ana_mean && rbf_mean < nn_mean) {
        printf "WINNER (Overall): RBF with %.4fm average error\n", rbf_mean
    } else {
        printf "WINNER (Overall): Analytical with %.4fm average error\n", ana_mean
    }
}
'
echo ""

# By robot configuration
echo "By Robot Configuration:"
echo ""

for series in 0XX 1XX 2XX; do
    echo "  $series:"
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        robot_type = $2
        ana_err = abs($47)  # analytical_vs_real
        nn_err = abs($48)   # nn_vs_real
        rbf_err = abs($49)  # rbf_vs_real

        ana_sum += ana_err
        nn_sum += nn_err
        rbf_sum += rbf_err

        ana_sq += ana_err * ana_err
        nn_sq += nn_err * nn_err
        rbf_sq += rbf_err * rbf_err

        count++
    }
    END {
        if (count > 0) {
            ana_mean = ana_sum / count
            nn_mean = nn_sum / count
            rbf_mean = rbf_sum / count

            ana_std = sqrt(ana_sq/count - ana_mean*ana_mean)
            nn_std = sqrt(nn_sq/count - nn_mean*nn_mean)
            rbf_std = sqrt(rbf_sq/count - rbf_mean*rbf_mean)

            printf "    Robot type: %s\n", robot_type
            printf "    Analytical:  %.4f ± %.4f m\n", ana_mean, ana_std
            printf "    Neural Net:  %.4f ± %.4f m\n", nn_mean, nn_std
            printf "    RBF:         %.4f ± %.4f m\n", rbf_mean, rbf_std

            if (nn_mean < ana_mean && nn_mean < rbf_mean) {
                printf "    WINNER: Neural Net (%.4fm)\n", nn_mean
            } else if (rbf_mean < ana_mean && rbf_mean < nn_mean) {
                printf "    WINNER: RBF (%.4fm)\n", rbf_mean
            } else {
                printf "    WINNER: Analytical (%.4fm)\n", ana_mean
            }
        }
    }
    '
    echo ""
done

echo "================================================================================"
echo "Q2: SIM-TO-REAL GAP vs APPROXIMATION ERROR"
echo "================================================================================"
echo ""

tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    ana_real += abs($47)    # analytical_vs_real (column 47)
    nn_ana += abs($45)      # nn_vs_analytical (column 45)
    rbf_ana += abs($46)     # rbf_vs_analytical (column 46)
    nn_real += abs($48)     # nn_vs_real (column 48)
    rbf_real += abs($49)    # rbf_vs_real (column 49)
    count++
}
END {
    reality_gap = ana_real / count
    nn_approx = nn_ana / count
    rbf_approx = rbf_ana / count
    nn_to_real = nn_real / count
    rbf_to_real = rbf_real / count

    printf "Comparing the '\''reality gap'\'' to ML approximation errors:\n\n"
    printf "Reality Gap (Analytical→Real):        %.4f m\n", reality_gap
    printf "  This is how far real robots deviate from perfect simulation\n\n"
    printf "NN Approximation Error (NN→Analytical):   %.4f m\n", nn_approx
    printf "RBF Approximation Error (RBF→Analytical): %.4f m\n\n", rbf_approx

    printf "Key Insight:\n"
    if (nn_approx < reality_gap && rbf_approx < reality_gap) {
        printf "  ✓ Both ML approximations are MORE accurate than the sim-to-real gap!\n"
        printf "    This means field approximation error < hardware/model mismatch\n"
    } else if (nn_approx > reality_gap || rbf_approx > reality_gap) {
        printf "  ✗ ML approximation errors EXCEED the sim-to-real gap\n"
        printf "    Field learning is the limiting factor, not hardware\n"
    } else {
        printf "  ~ ML approximation errors are similar to sim-to-real gap\n"
    }
    printf "\n"

    printf "Net Effect on Sim-to-Real Accuracy:\n"
    printf "  Using Analytical field: %.4fm error vs real\n", reality_gap
    printf "  Using NN field:         %.4fm error vs real\n", nn_to_real
    printf "  Using RBF field:        %.4fm error vs real\n\n", rbf_to_real

    if (nn_to_real < reality_gap) {
        improvement = reality_gap - nn_to_real
        pct = (improvement / reality_gap) * 100
        printf "  ✓ NN IMPROVES prediction by %.4fm (%.1f%%)!\n", improvement, pct
        printf "    (Learned field compensates for unmodeled dynamics)\n"
    } else if (nn_to_real > reality_gap) {
        degradation = nn_to_real - reality_gap
        pct = (degradation / reality_gap) * 100
        printf "  ✗ NN DEGRADES prediction by %.4fm (%.1f%%)\n", degradation, pct
        printf "    (Approximation error hurts more than it helps)\n"
    }

    if (rbf_to_real < reality_gap) {
        improvement = reality_gap - rbf_to_real
        pct = (improvement / reality_gap) * 100
        printf "  ✓ RBF IMPROVES prediction by %.4fm (%.1f%%)!\n", improvement, pct
    } else if (rbf_to_real > reality_gap) {
        degradation = rbf_to_real - reality_gap
        pct = (degradation / reality_gap) * 100
        printf "  ✗ RBF DEGRADES prediction by %.4fm (%.1f%%)\n", degradation, pct
    }
}
'
echo ""

echo "================================================================================"
echo "Q3: ADDITIONAL INSIGHTS"
echo "================================================================================"
echo ""

echo "INSIGHT 1: How does error scale with target radius?"
echo ""

for series in 0XX 1XX 2XX; do
    echo "  $series:"
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        robot_type = $2
        radius[NR] = $4
        ana_err[NR] = abs($47)  # analytical_vs_real
        nn_err[NR] = abs($48)   # nn_vs_real
        rbf_err[NR] = abs($49)  # rbf_vs_real
        n++
    }
    END {
        if (n > 1) {
            # Calculate means
            for (i=1; i<=n; i++) {
                r_sum += radius[i]
                ana_sum += ana_err[i]
                nn_sum += nn_err[i]
                rbf_sum += rbf_err[i]
            }
            r_mean = r_sum / n
            ana_mean = ana_sum / n
            nn_mean = nn_sum / n
            rbf_mean = rbf_sum / n

            # Calculate correlations
            for (i=1; i<=n; i++) {
                r_diff = radius[i] - r_mean

                ana_cov += r_diff * (ana_err[i] - ana_mean)
                nn_cov += r_diff * (nn_err[i] - nn_mean)
                rbf_cov += r_diff * (rbf_err[i] - rbf_mean)

                r_var += r_diff * r_diff
                ana_var += (ana_err[i] - ana_mean) * (ana_err[i] - ana_mean)
                nn_var += (nn_err[i] - nn_mean) * (nn_err[i] - nn_mean)
                rbf_var += (rbf_err[i] - rbf_mean) * (rbf_err[i] - rbf_mean)
            }

            ana_corr = ana_cov / sqrt(r_var * ana_var)
            nn_corr = nn_cov / sqrt(r_var * nn_var)
            rbf_corr = rbf_cov / sqrt(r_var * rbf_var)

            printf "    Robot type: %s\n", robot_type
            printf "    Correlation (radius vs error):\n"
            printf "      Analytical: %+.3f\n", ana_corr
            printf "      NN:         %+.3f\n", nn_corr
            printf "      RBF:        %+.3f\n", rbf_corr

            if (ana_corr < -0.3) {
                printf "    → Error DECREASES with radius (easier at large radii)\n"
            } else if (ana_corr > 0.3) {
                printf "    → Error INCREASES with radius (harder at large radii)\n"
            } else {
                printf "    → Error relatively independent of radius\n"
            }
        }
    }
    '
    echo ""
done

echo "INSIGHT 2: Center Estimation Accuracy"
echo ""
echo "Average distance of estimated center from true center (0,0):"
echo ""

for series in 0XX 1XX 2XX; do
    echo "  $series:"
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    $3 == s {
        robot_type = $2
        ana_center += $12   # analytical_avg_center_error
        nn_center += $22    # nn_avg_center_error
        rbf_center += $32   # rbf_avg_center_error
        real_center += $42  # real_avg_center_error
        count++
    }
    END {
        if (count > 0) {
            printf "    Robot type: %s\n", robot_type
            printf "    Analytical: %.4f m\n", ana_center/count
            printf "    NN:         %.4f m\n", nn_center/count
            printf "    RBF:        %.4f m\n", rbf_center/count
            printf "    Real:       %.4f m\n", real_center/count
        }
    }
    '
    echo ""
done

echo "INSIGHT 3: Which robot configuration performs best?"
echo ""

for series in 0XX 1XX 2XX; do
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        robot_type = $2
        real_err += abs($37)  # real_radius_error
        count++
    }
    END {
        if (count > 0) {
            printf "  %-25s: %.4fm average |radius error|\n", robot_type, real_err/count
        }
    }
    '
done
echo ""

echo "INSIGHT 4: Estimation Stability (std of center estimates over time)"
echo ""
echo "Lower values = more stable/consistent estimation"
echo ""

for series in 0XX 1XX 2XX; do
    echo "  $series:"
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    $3 == s {
        robot_type = $2
        ana_std += $13      # analytical_std_center_error
        nn_std += $23       # nn_std_center_error
        rbf_std += $33      # rbf_std_center_error
        real_std += $43     # real_std_center_error
        count++
    }
    END {
        if (count > 0) {
            ana_avg = ana_std / count
            nn_avg = nn_std / count
            rbf_avg = rbf_std / count
            real_avg = real_std / count

            printf "    Robot type: %s\n", robot_type
            printf "    Analytical: %.4f m\n", ana_avg
            printf "    NN:         %.4f m\n", nn_avg
            printf "    RBF:        %.4f m\n", rbf_avg
            printf "    Real:       %.4f m\n", real_avg

            if (nn_avg < ana_avg) {
                printf "    → NN estimates are MORE stable than analytical\n"
            }
            if (rbf_avg < ana_avg) {
                printf "    → RBF estimates are MORE stable than analytical\n"
            }
        }
    }
    '
    echo ""
done

echo "================================================================================"
echo "SUMMARY & RECOMMENDATIONS"
echo "================================================================================"
echo ""

# Get overall winner
WINNER=$(tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    ana_err += abs($47)  # analytical_vs_real
    nn_err += abs($48)   # nn_vs_real
    rbf_err += abs($49)  # rbf_vs_real
    count++
}
END {
    ana_mean = ana_err / count
    nn_mean = nn_err / count
    rbf_mean = rbf_err / count

    if (nn_mean < ana_mean && nn_mean < rbf_mean) {
        printf "NEURAL NETWORK (error: %.4fm)", nn_mean
    } else if (rbf_mean < ana_mean && rbf_mean < nn_mean) {
        printf "RBF (error: %.4fm)", rbf_mean
    } else {
        printf "ANALYTICAL (error: %.4fm)", ana_mean
    }
}
')

echo "1. BEST FIELD FOR REAL-WORLD PREDICTION:"
echo "   → Use $WINNER"
echo ""

# Field learning quality
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    nn_approx += abs($45)   # nn_vs_analytical
    rbf_approx += abs($46)  # rbf_vs_analytical
    count++
}
END {
    nn_avg = nn_approx / count
    rbf_avg = rbf_approx / count
    avg_ml = (nn_avg + rbf_avg) / 2

    printf "2. FIELD LEARNING QUALITY:\n"
    if (avg_ml < 0.02) {
        printf "   ✓ Excellent: ML models accurately approximate analytical field\n"
    } else if (avg_ml < 0.05) {
        printf "   ✓ Good: ML models reasonably approximate analytical field\n"
    } else {
        printf "   ✗ Poor: ML models struggle - consider retraining\n"
    }
    printf "   Average ML approximation error: %.4fm\n", avg_ml
}
'
echo ""

# Sim-to-real gap
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    reality_gap += abs($47)  # analytical_vs_real
    count++
}
END {
    gap = reality_gap / count
    printf "3. SIM-TO-REAL GAP:\n"
    if (gap < 0.05) {
        printf "   ✓ Excellent: Simulation closely matches reality\n"
    } else if (gap < 0.10) {
        printf "   ✓ Good: Reasonable sim-to-real transfer\n"
    } else {
        printf "   ⚠ Significant gap: Consider model calibration\n"
    }
    printf "   Reality gap: %.4fm\n", gap
}
'
echo ""

echo "================================================================================"
echo "ANALYSIS COMPLETE"
echo "================================================================================"
