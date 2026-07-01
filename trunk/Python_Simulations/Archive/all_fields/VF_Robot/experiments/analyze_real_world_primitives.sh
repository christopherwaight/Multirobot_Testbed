#!/bin/bash

CSV_FILE="sim2real_center_tracking_results/sim2real_center_tracking_results.csv"

echo "================================================================================"
echo "REAL WORLD PRIMITIVE PERFORMANCE ANALYSIS (Mean ± Std)"
echo "================================================================================"
echo ""

echo "Configuration → Primitive Mapping:"
echo "  0XX: critical_point_orbiter_plane_fitting (3-robot)"
echo "  1XX: center_orbiter_quad_planar (4-robot planar)"
echo "  2XX: center_orbiter_quad_advanced (4-robot dual Jacobian)"
echo ""

echo "================================================================================"
echo "OVERALL REAL ROBOT PERFORMANCE"
echo "================================================================================"
echo ""

for series in 0XX 1XX 2XX; do
    tail -n +2 "$CSV_FILE" | awk -F',' -v s="$series" '
    function abs(x) { return x < 0 ? -x : x }
    $3 == s {
        # Real radius error (column 37)
        real_err = abs($37)
        real_err_sum += real_err
        real_err_sq += real_err * real_err
        
        # Real std radius (column 36) - this is the variability within each run
        real_std_sum += $36
        
        count++
    }
    END {
        if (count > 0) {
            mean_err = real_err_sum / count
            std_err = sqrt(real_err_sq/count - mean_err*mean_err)
            avg_within_run_std = real_std_sum / count
            
            printf "%s:\n", s
            printf "  Mean radius error:           %.4f m\n", mean_err
            printf "  Std dev across configs:      %.4f m (variability between runs)\n", std_err
            printf "  Avg within-run std:          %.4f m (variability during orbit)\n", avg_within_run_std
            printf "  Range: [%.4f, %.4f] m\n\n", mean_err - std_err, mean_err + std_err
        }
    }
    '
done

echo "================================================================================"
echo "CONSISTENCY ANALYSIS"
echo "================================================================================"
echo ""

echo "Which primitive is most CONSISTENT (lowest variability)?"
echo ""

tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    series = $3
    real_err = abs($37)
    
    if (series == "0XX") {
        err_0XX[cnt_0XX] = real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX[cnt_1XX] = real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX[cnt_2XX] = real_err
        cnt_2XX++
    }
}
END {
    # Calculate mean and std for each
    for (i=0; i<cnt_0XX; i++) {
        sum_0XX += err_0XX[i]
    }
    mean_0XX = sum_0XX / cnt_0XX
    for (i=0; i<cnt_0XX; i++) {
        var_0XX += (err_0XX[i] - mean_0XX) * (err_0XX[i] - mean_0XX)
    }
    std_0XX = sqrt(var_0XX / cnt_0XX)
    cv_0XX = std_0XX / mean_0XX  # Coefficient of variation
    
    for (i=0; i<cnt_1XX; i++) {
        sum_1XX += err_1XX[i]
    }
    mean_1XX = sum_1XX / cnt_1XX
    for (i=0; i<cnt_1XX; i++) {
        var_1XX += (err_1XX[i] - mean_1XX) * (err_1XX[i] - mean_1XX)
    }
    std_1XX = sqrt(var_1XX / cnt_1XX)
    cv_1XX = std_1XX / mean_1XX
    
    for (i=0; i<cnt_2XX; i++) {
        sum_2XX += err_2XX[i]
    }
    mean_2XX = sum_2XX / cnt_2XX
    for (i=0; i<cnt_2XX; i++) {
        var_2XX += (err_2XX[i] - mean_2XX) * (err_2XX[i] - mean_2XX)
    }
    std_2XX = sqrt(var_2XX / cnt_2XX)
    cv_2XX = std_2XX / mean_2XX
    
    printf "0XX: std = %.4f m, CV = %.2f%% (higher = less consistent)\n", std_0XX, cv_0XX*100
    printf "1XX: std = %.4f m, CV = %.2f%%\n", std_1XX, cv_1XX*100
    printf "2XX: std = %.4f m, CV = %.2f%%\n\n", std_2XX, cv_2XX*100
    
    if (std_0XX < std_1XX && std_0XX < std_2XX) {
        printf "MOST CONSISTENT: 0XX (lowest std dev)\n"
    } else if (std_1XX < std_0XX && std_1XX < std_2XX) {
        printf "MOST CONSISTENT: 1XX (lowest std dev)\n"
    } else {
        printf "MOST CONSISTENT: 2XX (lowest std dev)\n"
    }
}
'
echo ""

echo "================================================================================"
echo "PERFORMANCE BY RADIUS (with std dev)"
echo "================================================================================"
echo ""

echo "Small Radius (0.01-0.20m):"
tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
$4 >= 0.01 && $4 <= 0.20 {
    series = $3
    real_err = abs($37)

    if (series == "0XX") {
        err_0XX += real_err
        sq_0XX += real_err * real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        sq_1XX += real_err * real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        sq_2XX += real_err * real_err
        cnt_2XX++
    }
}
END {
    if (cnt_0XX > 0) {
        mean = err_0XX/cnt_0XX
        std = sqrt(sq_0XX/cnt_0XX - mean*mean)
        printf "  0XX: %.4f ± %.4f m (n=%d)\n", mean, std, cnt_0XX
    }
    if (cnt_1XX > 0) {
        mean = err_1XX/cnt_1XX
        std = sqrt(sq_1XX/cnt_1XX - mean*mean)
        printf "  1XX: %.4f ± %.4f m (n=%d)\n", mean, std, cnt_1XX
    }
    if (cnt_2XX > 0) {
        mean = err_2XX/cnt_2XX
        std = sqrt(sq_2XX/cnt_2XX - mean*mean)
        printf "  2XX: %.4f ± %.4f m (n=%d)  ← ", mean, std, cnt_2XX
        if (mean < err_0XX/cnt_0XX && mean < err_1XX/cnt_1XX) printf "BEST\n"
        else printf "\n"
    }
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
        sq_0XX += real_err * real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        sq_1XX += real_err * real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        sq_2XX += real_err * real_err
        cnt_2XX++
    }
}
END {
    if (cnt_0XX > 0) {
        mean = err_0XX/cnt_0XX
        std = sqrt(sq_0XX/cnt_0XX - mean*mean)
        printf "  0XX: %.4f ± %.4f m (n=%d)  ← ", mean, std, cnt_0XX
        if (mean < err_1XX/cnt_1XX && mean < err_2XX/cnt_2XX) printf "BEST\n"
        else printf "\n"
    }
    if (cnt_1XX > 0) {
        mean = err_1XX/cnt_1XX
        std = sqrt(sq_1XX/cnt_1XX - mean*mean)
        printf "  1XX: %.4f ± %.4f m (n=%d)\n", mean, std, cnt_1XX
    }
    if (cnt_2XX > 0) {
        mean = err_2XX/cnt_2XX
        std = sqrt(sq_2XX/cnt_2XX - mean*mean)
        printf "  2XX: %.4f ± %.4f m (n=%d)\n", mean, std, cnt_2XX
    }
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
        sq_0XX += real_err * real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX += real_err
        sq_1XX += real_err * real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX += real_err
        sq_2XX += real_err * real_err
        cnt_2XX++
    }
}
END {
    if (cnt_0XX > 0) {
        mean = err_0XX/cnt_0XX
        std = sqrt(sq_0XX/cnt_0XX - mean*mean)
        printf "  0XX: %.4f ± %.4f m (n=%d)\n", mean, std, cnt_0XX
    }
    if (cnt_1XX > 0) {
        mean = err_1XX/cnt_1XX
        std = sqrt(sq_1XX/cnt_1XX - mean*mean)
        printf "  1XX: %.4f ± %.4f m (n=%d)  ← ", mean, std, cnt_1XX
        if (mean < err_0XX/cnt_0XX && mean < err_2XX/cnt_2XX) printf "BEST\n"
        else printf "\n"
    }
    if (cnt_2XX > 0) {
        mean = err_2XX/cnt_2XX
        std = sqrt(sq_2XX/cnt_2XX - mean*mean)
        printf "  2XX: %.4f ± %.4f m (n=%d)\n", mean, std, cnt_2XX
    }
}
'
echo ""

echo "================================================================================"
echo "RELIABILITY SCORE (Lower is Better)"
echo "================================================================================"
echo ""
echo "Score = Mean Error + Std Dev (penalizes both inaccuracy AND inconsistency)"
echo ""

tail -n +2 "$CSV_FILE" | awk -F',' '
function abs(x) { return x < 0 ? -x : x }
{
    series = $3
    real_err = abs($37)
    
    if (series == "0XX") {
        err_0XX[cnt_0XX] = real_err
        cnt_0XX++
    } else if (series == "1XX") {
        err_1XX[cnt_1XX] = real_err
        cnt_1XX++
    } else if (series == "2XX") {
        err_2XX[cnt_2XX] = real_err
        cnt_2XX++
    }
}
END {
    # 0XX
    for (i=0; i<cnt_0XX; i++) sum_0XX += err_0XX[i]
    mean_0XX = sum_0XX / cnt_0XX
    for (i=0; i<cnt_0XX; i++) var_0XX += (err_0XX[i] - mean_0XX) * (err_0XX[i] - mean_0XX)
    std_0XX = sqrt(var_0XX / cnt_0XX)
    score_0XX = mean_0XX + std_0XX
    
    # 1XX
    for (i=0; i<cnt_1XX; i++) sum_1XX += err_1XX[i]
    mean_1XX = sum_1XX / cnt_1XX
    for (i=0; i<cnt_1XX; i++) var_1XX += (err_1XX[i] - mean_1XX) * (err_1XX[i] - mean_1XX)
    std_1XX = sqrt(var_1XX / cnt_1XX)
    score_1XX = mean_1XX + std_1XX
    
    # 2XX
    for (i=0; i<cnt_2XX; i++) sum_2XX += err_2XX[i]
    mean_2XX = sum_2XX / cnt_2XX
    for (i=0; i<cnt_2XX; i++) var_2XX += (err_2XX[i] - mean_2XX) * (err_2XX[i] - mean_2XX)
    std_2XX = sqrt(var_2XX / cnt_2XX)
    score_2XX = mean_2XX + std_2XX
    
    printf "0XX: %.4f (mean) + %.4f (std) = %.4f\n", mean_0XX, std_0XX, score_0XX
    printf "1XX: %.4f (mean) + %.4f (std) = %.4f\n", mean_1XX, std_1XX, score_1XX
    printf "2XX: %.4f (mean) + %.4f (std) = %.4f\n\n", mean_2XX, std_2XX, score_2XX
    
    if (score_0XX < score_1XX && score_0XX < score_2XX) {
        printf "BEST RELIABILITY: 0XX\n"
    } else if (score_1XX < score_0XX && score_1XX < score_2XX) {
        printf "BEST RELIABILITY: 1XX\n"
    } else {
        printf "BEST RELIABILITY: 2XX  ← WINNER\n"
    }
}
'
echo ""

echo "================================================================================"
echo "FINAL VERDICT"
echo "================================================================================"
