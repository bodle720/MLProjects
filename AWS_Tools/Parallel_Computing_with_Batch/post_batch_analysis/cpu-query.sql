fields @timestamp, @message 
| filter @message like /\[CPU_METRIC\] peak_cpu_pct=/ 
| parse @message "[CPU_METRIC] peak_cpu_pct=*, avg_cpu_pct=*" as peak_cpu, avg_cpu 
| stats max(peak_cpu) as max_peak_cpu, min(peak_cpu) as min_peak_cpu, avg(peak_cpu) as avg_peak_cpu, max(avg_cpu) as max_avg_cpu, min(avg_cpu) as min_avg_cpu, avg(avg_cpu) as avg_avg_cpu, count(@logStream) as count