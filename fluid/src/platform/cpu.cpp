#include "cpu.h"

namespace padle_mobile {
namespace platform {

int getCpuCount() {
#ifdef PLATFORM_ANDROID
  // get cpu count from /proc/cpuinfo
  FILE *fp = fopen("/proc/cpuinfo", "rb");
  if (!fp) {
    return 1;
  }

  int count = 0;
  char line[1024];
  while (!feof(fp)) {
    char *s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }

    if (memcmp(line, "processor", 9) == 0) {
      count++;
    }
  }

  fclose(fp);

  if (count < 1) {
    count = 1;
  }

  return count;
#elif __IOS__
  int count = 0;
  size_t len = sizeof(count);
  sysctlbyname("hw.ncpu", &count, &len, NULL, 0);

  if (count < 1) {
    count = 1;
  }

  return count;
#else
  return 1;
#endif
}

int getMemInfo() {
#ifdef PLATFORM_ANDROID
  // get cpu count from /proc/cpuinfo
  FILE *fp = fopen("/proc/meminfo", "rb");
  if (!fp) {
    return 1;
  }

  int memsize = 0;
  char line[1024];
  while (!feof(fp)) {
    char *s = fgets(line, 1024, fp);
    if (!s) {
      break;
    }
    sscanf(s, "MemTotal:        %d kB", &memsize);
  }

  fclose(fp);

  return memsize;
#elif __IOS__
  // to be implemented
  return 0;
#endif
}

int getMaxFreq(int cpuid) {
  // first try, for all possible cpu
  char path[256];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

  FILE *fp = fopen(path, "rb");

  if (!fp) {
    // second try, for online cpu
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state",
             cpuid);
    fp = fopen(path, "rb");

    if (!fp) {
      // third try, for online cpu
      snprintf(path, sizeof(path),
               "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
      fp = fopen(path, "rb");

      if (!fp) {
        return -1;
      }

      int max_freq_khz = -1;
      fscanf(fp, "%d", &max_freq_khz);

      fclose(fp);

      return max_freq_khz;
    }
  }

  int max_freq_khz = 0;
  while (!feof(fp)) {
    int freq_khz = 0;
    int nscan = fscanf(fp, "%d %*d", &freq_khz);
    if (nscan != 1) {
      break;
    }

    if (freq_khz > max_freq_khz) {
      max_freq_khz = freq_khz;
    }
  }

  fclose(fp);

  return max_freq_khz;
}

int sortBigLittleByFreq(int cpuCount, std::vector<int> &cpuIds,
                        std::vector<int> &cpuFreq,
                        std::vector<int> &clusterIds) {
  // const int cpuCount = cpuIds.size();

  if (cpuCount == 0) {
    return 0;
  }

  // std::vector<int> cpu_max_freq_khz;
  cpuIds.resize(cpuCount);
  cpuFreq.resize(cpuCount);
  clusterIds.resize(cpuCount);

  for (int i = 0; i < cpuCount; i++) {
    int max_freq_khz = getMaxFreq(i);
    // printf("%d max freq = %d khz\n", i, max_freq_khz);
    cpuIds[i] = i;
    cpuFreq[i] = max_freq_khz / 1000;
  }

  // sort cpuid as big core first
  // simple bubble sort
  /*
  for (int i = 0; i < cpuCount; i++)
  {
      for (int j = i+1; j < cpuCount; j++)
      {
          if (cpuFreq[i] < cpuFreq[j])
          {
              // swap
              int tmp = cpuIds[i];
              cpuIds[i] = cpuIds[j];
              cpuIds[j] = tmp;

              tmp = cpuFreq[i];
              cpuFreq[i] = cpuFreq[j];
              cpuFreq[j] = tmp;
          }
      }
  }*/

  // SMP
  int mid_max_freq_khz = (cpuFreq.front() + cpuFreq.back()) / 2;
  // if (mid_max_freq_khz == cpuFreq.back())
  //    return 0;

  for (int i = 0; i < cpuCount; i++) {
    if (cpuFreq[i] >= mid_max_freq_khz) {
      clusterIds[i] = 0;
    } else {
      clusterIds[i] = 1;
    }
  }

  return 0;
}

#ifdef PLATFORM_ANDROID
int setSchedAffinity(const std::vector<int> &cpuIds) {
  // cpu_set_t definition
  // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity

  typedef struct {
    unsigned long mask_bits[1024 / __NCPUBITS__];
  } cpu_set_t;

  // set affinity for thread
  pid_t pid = gettid();

  cpu_set_t mask;
  __CPU_ZERO(&mask);
  for (int i = 0; i < (int)cpuIds.size(); i++) {
    __CPU_SET(cpuIds[i], &mask);
  }

  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallret) {
    std::cout << "syscall error " << syscallret;
    return -1;
  }

  return 0;
}

int setCpuAffinity(const std::vector<int> &cpuIds) {
#ifdef USE_OPENMP
  int num_threads = cpuIds.size();
  omp_set_num_threads(num_threads);
  std::vector<int> ssarets(num_threads, 0);
#pragma omp parallel for
  for (int i = 0; i < num_threads; i++) {
    ssarets[i] = setSchedAffinity(cpuIds);
  }
  for (int i = 0; i < num_threads; i++) {
    if (ssarets[i] != 0) {
      std::cout << "set cpu affinity failed, cpuID: " << cpuIds[i];
      return -1;
    }
  }
#else
  std::vector<int> cpuid1;
  cpuid1.push_back(cpuIds[0]);
  int ssaret = setSchedAffinity(cpuid1);
  if (ssaret != 0) {
    std::cout << "set cpu affinity failed, cpuID: " << cpuIds[0];
    return -1;
  }
#endif
}
#endif

}  // namespace platform
}  // namespace padle_mobile
