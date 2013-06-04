/*
 * Wavelet transforms.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


/* Daubechies coefficients */
static const double Daub2f[] = {
    7.07106781186547524e-01f,
    7.07106781186547524e-01f
};

static const double Daub4f[] = {
    4.82962913144534143e-01f,
    8.36516303737807906e-01f,
    2.24143868042013381e-01f,
   -1.29409522551260381e-01f
};

static const double Daub6f[] = {
    3.32670552950082616e-01f,
    8.06891509311092576e-01f,
    4.59877502118491570e-01f,
   -1.35011020010254589e-01f,
   -8.54412738820266617e-02f,
    3.52262918857095366e-02f
};

static const double Daub8f[] = {
    2.30377813308896501e-01f,
    7.14846570552915647e-01f,
    6.30880767929858908e-01f,
   -2.79837694168598542e-02f,
   -1.87034811719093084e-01f,
    3.08413818355607636e-02f,
    3.28830116668851997e-02f,
   -1.05974017850690321e-02f
};

static const double Daub10f[] = {
    1.60102397974192914e-01f,
    6.03829269797189671e-01f,
    7.24308528437772928e-01f,
    1.38428145901320732e-01f,
   -2.42294887066382032e-01f,
   -3.22448695846383746e-02f,
    7.75714938400457135e-02f,
   -6.24149021279827427e-03f,
   -1.25807519990819995e-02f,
    3.33572528547377128e-03f
};

static const double *DaubLf[] = {
    Daub2f, Daub4f, Daub6f, Daub8f, Daub10f
};


/* wavelet transform pass (horizontal or vertical) */
static void transform_pass(int L, float *in, float *out, int width, int height, int row_width, float *lowpass, float *highpass)
{
 int hh = (height >> 1) * row_width;
 int row, column;
 int i;

 for(row = 0; row < height; row++, in += row_width)
 {
  int offs = 0;
  for(column = 0; column < width - L; column += 2, offs += row_width)
  {
   float c = 0, d = 0;
   for(i = 0; i < L; i++)
   {
    c += lowpass[i] * in[column + i];
    d += highpass[i] * in[column + i];
   }
   // output to column number row
   // here offs = offset to row in texture
   out[row + offs] = c;
   out[row + offs + hh] = d;
  }
  // calculate values at border
  for( ; column < width; column += 2, offs += row_width)
  {
   float c = 0, d = 0;
   for(i = 0; i < L; i++)
   {
    c += lowpass[i] * in[(column + i) & (width - 1)]; // width is a power of 2
    d += highpass[i] * in[(column + i) & (width - 1)]; // width is a power of 2
   }
   // output to column number row
   // here offs = offset to row in texture
   out[row + offs] = c;
   out[row + offs + hh] = d;
  }
 }
}


/* wavelet transform reverse pass (vertical or horizontal) */
static void reverse_pass(int L, float *in, float *out, int width, int height, int row_width, float *lowpass, float *highpass)
{
 int row, column;
 int i, j;

 width >>= 1;
 for(row = 0; row < height; row++, in += row_width, out++)
 {
  int offs = 0;
  for(column = width - ((L >> 1) - 1); column < (width << 1) - ((L >> 1) - 1); column++, offs += row_width << 1)
  {
   float a = 0, b = 0;
   for(i = L - 2, j = 0; i >= 0; i -= 2, j++)
   {
    a += lowpass[i] * in[(column + j) & (width - 1)] + highpass[i] * in[width + ((column + j) & (width - 1))];
    b += lowpass[i + 1] * in[(column + j) & (width - 1)] + highpass[i + 1] * in[width + ((column + j) & (width - 1))];
   }
   // output to column number row
   // here offs = offset to row in texture
   out[offs] = a;
   out[offs + row_width] = b;
  }
 }
}


/* transpose the texture */
static void transpose(float *in, float *out, int width, int height)
{
 int row, column, offs;

 for(row = 0; row < height; row++, out++, in += width)
 {
  for(column = 0, offs = 0; column < width; column++, offs += width)
  {
   out[offs] = in[column];
  }
 }
}


/* update texture */
static void UpdateTexture(int L, int N, float *texture, float *texture_temp, int texture_width, int texture_height, int forward_pass)
{
 float lowpass[L];
 float highpass[L];

 int i;
 for(i = 0; i < L; i++)
 {
  lowpass[i] = DaubLf[(L >> 1) - 1][i];
  highpass[L - 1 - i] = (i & 1) ? -lowpass[i] : lowpass[i];
 }

 int width, height;

 if(forward_pass)
 {
  width = texture_width;
  height = texture_height;

  while((width > N) && (height > N))
  {
   // horizontal pass
   transform_pass(L, texture, texture_temp, width, height, texture_width, lowpass, highpass);

   // vertical pass (simply swap width and height)
   transform_pass(L, texture_temp, texture, height, width, texture_height, lowpass, highpass);

   width >>= 1;
   height >>= 1;
  }
 }
 else
 {
  transpose(texture, texture_temp, texture_width, texture_height);

  width = N << 1;
  height = N << 1;

  while((width <= texture_width) && (height <= texture_height))
  {
   // vertical pass
   reverse_pass(L, texture_temp, texture, height, width, texture_height, lowpass, highpass);

   // horizontal pass (simply swap height and width)
   reverse_pass(L, texture, texture_temp, width, height, texture_width, lowpass, highpass);

   width <<= 1;
   height <<= 1;
  }

  transpose(texture_temp, texture, texture_height, texture_width);
 }
}


static int transform(float *a, float *tmp, int width, int height, int N, int L, int forward)
{
 if(N < 0)
  N = 0;
 if(L & 1)
  L++;
 if(L <= 0)
  L = 2;
 if(L > 10)
  L = 10;
 UpdateTexture(L, N, a, tmp, width, height, forward);
 return 0;
}
