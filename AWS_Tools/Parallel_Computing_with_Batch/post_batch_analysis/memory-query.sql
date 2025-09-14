fields @timestamp, @message
| filter @message like /\[MEMORY_METRIC\] peak_memory_mib=/
| parse @message "[MEMORY_METRIC] peak_memory_mib=*" as peak_memory
| stats max(peak_memory) as max_memory, min(peak_memory) as min_memory, avg(peak_memory) as avg_memory, count(@logStream) as count
