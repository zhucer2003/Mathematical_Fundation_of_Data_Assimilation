function rng(seed)
%RNG MATLABの組み込み関数rngのseedによる初期化部分のみの互換関数
%   Octaveにrngが存在しないため，rngのseedを使用した乱数初期化部分の互換関数
% Author: Masahide Abe (Tohoku Univ.)
% Version: 1.0 (2014/5/8)
randn('seed', seed)
end