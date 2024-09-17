template <typename T, int N>
struct RegisterIndexUtils {
  __device__ __forceinline__ static T get(const T arr[N], int idx) {
    if (idx < 0 || idx >= N)
      return T();
    switch (idx) {
      case 0:
        return arr[0];
      case 1:
        return arr[1];
      case 2:
        return arr[2];
      case 3:
        return arr[3];
      case 4:
        return arr[4];
      case 5:
        return arr[5];
      case 6:
        return arr[6];
      case 7:
        return arr[7];
      case 8:
        return arr[8];
      case 9:
        return arr[9];
      case 10:
        return arr[10];
      case 11:
        return arr[11];
      case 12:
        return arr[12];
      case 13:
        return arr[13];
      case 14:
        return arr[14];
      case 15:
        return arr[15];
      case 16:
        return arr[16];
      case 17:
        return arr[17];
      case 18:
        return arr[18];
      case 19:
        return arr[19];
      case 20:
        return arr[20];
      case 21:
        return arr[21];
      case 22:
        return arr[22];
      case 23:
        return arr[23];
      case 24:
        return arr[24];
      case 25:
        return arr[25];
      case 26:
        return arr[26];
      case 27:
        return arr[27];
      case 28:
        return arr[28];
      case 29:
        return arr[29];
      case 30:
        return arr[30];
      case 31:
        return arr[31];
    };
    return T();
  }

  __device__ __forceinline__ static void set(T arr[N], int idx, T val) {
    if (idx < 0 || idx >= N)
      return;
    switch (idx) {
      case 0:
        arr[0] = val;
        break;
      case 1:
        arr[1] = val;
        break;
      case 2:
        arr[2] = val;
        break;
      case 3:
        arr[3] = val;
        break;
      case 4:
        arr[4] = val;
        break;
      case 5:
        arr[5] = val;
        break;
      case 6:
        arr[6] = val;
        break;
      case 7:
        arr[7] = val;
        break;
      case 8:
        arr[8] = val;
        break;
      case 9:
        arr[9] = val;
        break;
      case 10:
        arr[10] = val;
        break;
      case 11:
        arr[11] = val;
        break;
      case 12:
        arr[12] = val;
        break;
      case 13:
        arr[13] = val;
        break;
      case 14:
        arr[14] = val;
        break;
      case 15:
        arr[15] = val;
        break;
      case 16:
        arr[16] = val;
        break;
      case 17:
        arr[17] = val;
        break;
      case 18:
        arr[18] = val;
        break;
      case 19:
        arr[19] = val;
        break;
      case 20:
        arr[20] = val;
        break;
      case 21:
        arr[21] = val;
        break;
      case 22:
        arr[22] = val;
        break;
      case 23:
        arr[23] = val;
        break;
      case 24:
        arr[24] = val;
        break;
      case 25:
        arr[25] = val;
        break;
      case 26:
        arr[26] = val;
        break;
      case 27:
        arr[27] = val;
        break;
      case 28:
        arr[28] = val;
        break;
      case 29:
        arr[29] = val;
        break;
      case 30:
        arr[30] = val;
        break;
      case 31:
        arr[31] = val;
        break;
    }
  }
};
