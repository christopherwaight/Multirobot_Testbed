#!/bin/bash

# Analyze control primitive performance in real world and NN simulation

CSV_FILE="sim2real_center_tracking_results/sim2real_center_tracking_results.csv"

echo "================================================================================"
echo "CONTROL PRIMITIVE PERFORMANCE ANALYSIS"
echo "================================================================================"
echo ""

echo "Configuration → Primitive Mapping:"
echo "  0XX (3-robot):          critical_point_orbiter_plane_fitting"
echo "  1XX (4-robot-square):   center_orbiter_quad_planar"
echo "  2XX (4-robot-advanced): center_orbiter_quad_advanced"
echo ""

echo "================================================================================"
echo "Q1: WHICH PRIMITIVE WORKS BEST IN REAL WORLD?"
echo "================================================================================"
echo ""

echo "Real Robot Performance (absolute radius error):"
echo ""

for series in 0XX 1XX 2XX; do
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        robot_type = $2
        real_err += abs($37)  # real_radius_error
        real_sq += $37 * $37
        count++
    }
    END {
        if (count > 0) {
            mean = real_err / count
            std = sqrt(real_sq/count - (real_err/count)*(real_err/count))
            printf "  %s: %.4f ± %.4f m\n", s, mean, std
        }
    }
    '
done

echo ""
echo "WINNER: Best real-world primitive"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    series = $3
    real_err = abs($37)

    if (series == "0XX") {
        err_0XX += real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        cnt_2XX++
    }
}
END {
    mean_0XX = err_0XX / cnt_0XX
    mean_1XX = err_1XX / cnt_1XX
    mean_2XX = err_2XX / cnt_2XX

    if (mean_0XX < mean_1XX && mean_0XX < mean_2XX) {
        printf "  → 0XX (critical_point_orbiter_plane_fitting): %.4fm\n", mean_0XX
        printf "     3-robot plane fitting orbiter\n"
    } else if (mean_1XX < mean_0XX && mean_1XX < mean_2XX) {
        printf "  → 1XX (center_orbiter_quad_planar): %.4fm\n", mean_1XX
        printf "     4-robot planar fitting orbiter\n"
    } else {
        printf "  → 2XX (center_orbiter_quad_advanced): %.4fm\n", mean_2XX
        printf "     4-robot dual Jacobian orbiter\n"
    }
}
'
echo ""

echo "================================================================================"
echo "Q2: HOW WELL DOES NN PREDICT REAL WORLD FOR EACH PRIMITIVE?"
echo "================================================================================"
echo ""

echo "NN Simulation vs Real Robot Error (nn_vs_real):"
echo ""

for series in 0XX 1XX 2XX; do
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        robot_type = $2
        nn_real_err += abs($48)  # nn_vs_real
        nn_sq += $48 * $48
        count++
    }
    END {
        if (count > 0) {
            mean = nn_real_err / count
            std = sqrt(nn_sq/count - (nn_real_err/count)*(nn_real_err/count))
            printf "  %s: %.4f ± %.4f m\n", s, mean, std
        }
    }
    '
done

echo ""
echo "Interpretation:"
echo "  Lower error = NN field better predicts real-world primitive behavior"
echo ""

echo "BEST NN PREDICTION:"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    series = $3
    nn_real = abs($48)

    if (series == "0XX") {
        err_0XX += nn_real
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += nn_real
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += nn_real
        cnt_2XX++
    }
}
END {
    mean_0XX = err_0XX / cnt_0XX
    mean_1XX = err_1XX / cnt_1XX
    mean_2XX = err_2XX / cnt_2XX

    if (mean_0XX < mean_1XX && mean_0XX < mean_2XX) {
        printf "  → 0XX (critical_point_orbiter_plane_fitting)\n"
        printf "     NN error: %.4fm - Most predictable primitive\n", mean_0XX
    } else if (mean_1XX < mean_0XX && mean_1XX < mean_2XX) {
        printf "  → 1XX (center_orbiter_quad_planar)\n"
        printf "     NN error: %.4fm - Most predictable primitive\n", mean_1XX
    } else {
        printf "  → 2XX (center_orbiter_quad_advanced)\n"
        printf "     NN error: %.4fm - Most predictable primitive\n", mean_2XX
    }
}
'
echo ""

echo "================================================================================"
echo "Q3: DETAILED PRIMITIVE COMPARISON"
echo "================================================================================"
echo ""

for series in 0XX 1XX 2XX; do
    if [ "$series" = "0XX" ]; then
        prim_name="critical_point_orbiter_plane_fitting"
        prim_desc="3-robot plane fitting orbiter"
    elif [ "$series" = "1XX" ]; then
        prim_name="center_orbiter_quad_planar"
        prim_desc="4-robot planar (4-point) fitting orbiter"
    else
        prim_name="center_orbiter_quad_advanced"
        prim_desc="4-robot dual Jacobian orbiter"
    fi

    echo "───────────────────────────────────────────────────────────────────────────"
    echo "PRIMITIVE: $prim_name"
    echo "Description: $prim_desc"
    echo "───────────────────────────────────────────────────────────────────────────"
    echo ""

    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        # Real robot performance
        real_err += abs($37)
        real_rmse += $44 * $44
        real_center_err += $42
        real_center_std += $43

        # NN performance
        nn_err += abs($17)
        nn_rmse += $24 * $24
        nn_center_err += $22
        nn_center_std += $23

        # NN vs Real gap
        nn_vs_real += abs($48)

        # Analytical vs Real gap
        ana_vs_real += abs($47)

        count++
    }
    END {
        if (count > 0) {
            printf "Real Robot Performance:\n"
            printf "  Average radius error:     %.4f m\n", real_err/count
            printf "  RMSE:                     %.4f m\n", sqrt(real_rmse/count)
            printf "  Center estimation error:  %.4f m\n", real_center_err/count
            printf "  Center estimation std:    %.4f m\n\n", real_center_std/count

            printf "NN Simulation Performance:\n"
            printf "  Average radius error:     %.4f m\n", nn_err/count
            printf "  RMSE:                     %.4f m\n", sqrt(nn_rmse/count)
            printf "  Center estimation error:  %.4f m\n", nn_center_err/count
            printf "  Center estimation std:    %.4f m\n\n", nn_center_std/count

            printf "Prediction Accuracy:\n"
            printf "  NN→Real error:            %.4f m\n", nn_vs_real/count
            printf "  Analytical→Real error:    %.4f m\n\n", ana_vs_real/count

            improvement = ana_vs_real/count - nn_vs_real/count
            if (improvement > 0) {
                pct = (improvement / (ana_vs_real/count)) * 100
                printf "  ✓ NN IMPROVES real-world prediction by %.4fm (%.1f%%)\n", improvement, pct
            } else {
                pct = (-improvement / (ana_vs_real/count)) * 100
                printf "  ✗ NN DEGRADES real-world prediction by %.4fm (%.1f%%)\n", -improvement, pct
            }
        }
    }
    '
    echo ""
done

echo "================================================================================"
echo "Q4: WHICH PRIMITIVE WORKS BEST AND WHY?"
echo "================================================================================"
echo ""

echo "Summary by Target Radius:"
echo ""

# Group by radius ranges
echo "Small Radius (0.01-0.20m):"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
$4 >= 0.01 && $4 <= 0.20 {
    series = $3
    real_err = abs($37)

    if (series == "0XX") {
        err_0XX += real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        cnt_2XX++
    }
}
END {
    if (cnt_0XX > 0) printf "  0XX (plane_fitting):      %.4f m (n=%d)\n", err_0XX/cnt_0XX, cnt_0XX
    if (cnt_1XX > 0) printf "  1XX (quad_planar):        %.4f m (n=%d)\n", err_1XX/cnt_1XX, cnt_1XX
    if (cnt_2XX > 0) printf "  2XX (dual_jacobian):      %.4f m (n=%d)\n", err_2XX/cnt_2XX, cnt_2XX
}
'
echo ""

echo "Medium Radius (0.21-0.40m):"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
$4 > 0.20 && $4 <= 0.40 {
    series = $3
    real_err = abs($37)

    if (series == "0XX") {
        err_0XX += real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        cnt_2XX++
    }
}
END {
    if (cnt_0XX > 0) printf "  0XX (plane_fitting):      %.4f m (n=%d)\n", err_0XX/cnt_0XX, cnt_0XX
    if (cnt_1XX > 0) printf "  1XX (quad_planar):        %.4f m (n=%d)\n", err_1XX/cnt_1XX, cnt_1XX
    if (cnt_2XX > 0) printf "  2XX (dual_jacobian):      %.4f m (n=%d)\n", err_2XX/cnt_2XX, cnt_2XX
}
'
echo ""

echo "Large Radius (0.41+m):"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
$4 > 0.40 {
    series = $3
    real_err = abs($37)

    if (series == "0XX") {
        err_0XX += real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        cnt_2XX++
    }
}
END {
    if (cnt_0XX > 0) printf "  0XX (plane_fitting):      %.4f m (n=%d)\n", err_0XX/cnt_0XX, cnt_0XX
    if (cnt_1XX > 0) printf "  1XX (quad_planar):        %.4f m (n=%d)\n", err_1XX/cnt_1XX, cnt_1XX
    if (cnt_2XX > 0) printf "  2XX (dual_jacobian):      %.4f m (n=%d)\n", err_2XX/cnt_2XX, cnt_2XX
}
'
echo ""

echo "================================================================================"
echo "FINAL RECOMMENDATIONS"
echo "================================================================================"
echo ""

# Get overall best primitive
BEST_PRIM=$(tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    series = $3
    real_err = abs($37)

    if (series == "0XX") {
        err_0XX += real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        cnt_2XX++
    }
}
END {
    mean_0XX = err_0XX / cnt_0XX
    mean_1XX = err_1XX / cnt_1XX
    mean_2XX = err_2XX / cnt_2XX

    if (mean_0XX < mean_1XX && mean_0XX < mean_2XX) {
        print "critical_point_orbiter_plane_fitting (3-robot)"
    } else if (mean_1XX < mean_0XX && mean_1XX < mean_2XX) {
        print "center_orbiter_quad_planar (4-robot-square)"
    } else {
        print "center_orbiter_quad_advanced (4-robot-advanced)"
    }
}
')

echo "1. BEST REAL-WORLD PRIMITIVE:"
echo "   → $BEST_PRIM"
echo ""

echo "2. NN FIELD PREDICTION QUALITY:"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    series = $3
    nn_real = abs($48)
    ana_real = abs($47)

    nn_total += nn_real
    ana_total += ana_real
    count++
}
END {
    nn_avg = nn_total / count
    ana_avg = ana_total / count

    if (nn_avg < ana_avg) {
        improvement = ana_avg - nn_avg
        pct = (improvement / ana_avg) * 100
        printf "   ✓ NN fields improve prediction by %.1f%% across all primitives\n", pct
        printf "   NN understands real robot dynamics better than analytical model\n"
    } else {
        printf "   ✗ Analytical model predicts better than NN\n"
    }
}
'
echo ""

echo "3. WHEN TO USE EACH PRIMITIVE:"
echo ""
echo "   → Use critical_point_orbiter_plane_fitting (3-robot) when:"
echo "     • You have exactly 3 robots available"
echo "     • You need simple, exact solution (3 points = unique plane)"
echo "     • Formation can be maintained as equilateral triangle"
echo ""
echo "   → Use center_orbiter_quad_planar (4-robot) when:"
echo "     • You have 4 robots in square/rectangular formation"
echo "     • You want redundancy (overdetermined 4-point fit)"
echo "     • Medium accuracy requirements"
echo ""
echo "   → Use center_orbiter_quad_advanced (4-robot) when:"
echo "     • You have 4 robots in advanced formation (2 triangles)"
echo "     • You need best accuracy and robustness"
echo "     • Can maintain quadrilateral with two overlapping triangles"
echo ""

echo "================================================================================"
echo "ANALYSIS COMPLETE"
echo "================================================================================"
