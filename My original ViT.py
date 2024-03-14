#VisionTransformer入門 Conputer Vision Libraryに記載のVitのアーキテクチャ実装
import torch
import torch.nn as nn

import torchvision

#Get dataset CIFAR100
#cifar100_data=transvision.dataset.CIFAR100('./cifar-100',train=True,download=True,transform=torchvision.transforms.ToTensor())
#data_loader=torch.utils.DataLoader(cifar100_data,batch_size=4,shuffle=True)

class Vit(nn.Module):
    def __init__(self,
        in_channels:int=3,
        num_classes:int=10,
        emb_dim:int=384,
        num_patch_row:int=2,
        image_size:int=32,
        num_blocks:int=7,
        head:int=8,
        hidden_dim:int=384*4,
        dropout:float=0.
        ):
        """
         引数:
             in_channels: 入力画像のチャンネル数
             num_classes: 画像分類のクラス数
             emb_dim: 埋め込み後のベクトルの長さ
             num_patch_row: 1辺のパッチの数
             image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
             num_blocks: Encoder Blockの数
             head: ヘッドの数
             hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ
             dropout: ドロップアウト率
        """
        super(Vit, self).__init__()
        # Input Layer [2-3節]
        self.input_layer = VitInputLayer(
            in_channels,
            emb_dim,
            num_patch_row,
            image_size)


        # Encoder。Encoder Blockの多段。[2-5節]
        self.encoder = nn.Sequential(*[
             VitEncoderBlock(
                 emb_dim=emb_dim,
                 head=head,
                 hidden_dim=hidden_dim,
                 dropout = dropout
             )
             for _ in range(num_blocks)])


        # MLP Head [2-6-1項]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: ViTへの入力画像。形状は、(B, C, H, W)
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            out: ViTの出力。形状は、(B, M)。[式(10)]
                B:バッチサイズ、M:クラス数
        """
        # Input Layer [式(14)]
        ## (B, C, H, W) -> (B, N, D)
        ## N: トークン数(=パッチの数+1), D: ベクトルの長さ
        out = self.input_layer(x)

        # Encoder [式(15)、式(16)]
        ## (B, N, D) -> (B, N, D)
        out = self.encoder(out)
        # クラストークンのみ抜き出す
        ## (B, N, D) -> (B, D)
        cls_token = out[:,0]
        # MLP Head [式(17)]
        ## (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred


#***************************************************************
#VitInputLayerのclass定義(p.85参照)

class VitInputLayer(nn.Module):
    def __int__(self,in_channels:int=3,emb_dim:int=384,
    num_patch_row:int=2,image_size:int=32):

#         引数:
#             in_channels: 入力画像のチャンネル数
#             emb_dim: 埋め込み後のベクトルの長さ
#             num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした
#             image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定

        super(VitInputLayer, self).__init__()
        self.in_channels=in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        # パッチの数
        ## 例: 入力画像を2x2のパッチに分ける場合、num_patchは4
        self.num_patch = self.num_patch_row**2

        # パッチの大きさ
        ## 例: 入力画像の1辺の大きさが32の場合、patch_sizeは16
        self.patch_size = int(self.image_size // self.num_patch_row)
        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        # 位置埋め込み
        ## クラストークンが先頭に結合されているため、
        ## 長さemb_dimの位置埋め込みベクトルを(パッチ数+1)個用意
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim)
        )


#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#
#    引数:
#             x: 入力画像。形状は、(B, C, H, W)。[式(1)]
#             B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
#
#    返り値:
#             z_0: ViTへの入力。形状は、(B, N, D)。
#             B:バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ


    # パッチの埋め込み & flatten [式(3)]
    ## パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
    ## ここで、Pはパッチ1辺の大きさ
        z_0 = self.patch_emb_layer(x)

    ## パッチのflatten (B, D, H/P, W/P) -> (B, D, Np)
    ## ここで、Npはパッチの数(=H*W/P^2)
        z_0 = z_0.flatten(2)

    ##  軸の入れ替え (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1, 2)

    # パッチの埋め込みの先頭にクラストークンを結合 [式(4)]
    ## (B, Np, D) -> (B, N, D)
    ## N = (Np + 1)であることに留意
    ## また、cls_tokenの形状は(1,1,D)であるため、
        z_0 = self.patch_emb_layer(x)

    ## パッチのflatten (B, D, H/P, W/P) -> (B, D, Np)
    ## ここで、Npはパッチの数(=H*W/P^2)
        z_0 = z_0.flatten(2)

    ##  軸の入れ替え (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1, 2)

    # パッチの埋め込みの先頭にクラストークンを結合 [式(4)]
    ## (B, Np, D) -> (B, N, D)
    ## N = (Np + 1)であることに留意
    ## また、cls_tokenの形状は(1,1,D)であるため、
    ## repeatメソッドによって(B,1,D)に変換してからパッチの埋め込みとの結合を行う
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)

    # 位置埋め込みの加算 [式(5)]
    ## (B, N, D) -> (B, N, D)
        z_0 = z_0 + self.pos_emb

        return z_0


#*********************************************************************
#Layer Normalizationのプログラムコードは下記Encoder Block参照
#*********************************************************************
#MLPのプログラムコードは下記Encoder Block参照
#*********************************************************************
#Encoder Blockのプログラムコードは


class VitEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim:int=384,
        head:int=8,
        hidden_dim:int=384*4,
        dropout: float=0.
        ):

#    引数:
#          emb_dim: 埋め込み後のベクトルの長さ
#          head: ヘッドの数
#          hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ
#          原論文に従ってemb_dimの4倍をデフォルト値としている
#          dropout: ドロップアウト率

      super(VitEncoderBlock, self).__init__()
      # 1つ目のLayer Normalization [2-5-2項]
      self.ln1 = nn.LayerNorm(emb_dim)
      # MHSA [2-4-7項]
      self.msa = MultiHeadSelfAttention(
          emb_dim=emb_dim,
          head=head,
          dropout = dropout,
        )
    # 2つ目のLayer Normalization [2-5-2項]
      self.ln2 = nn.LayerNorm(emb_dim)
      # MLP [2-5-3項]
      self.mlp = nn.Sequential(
        nn.Linear(emb_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, emb_dim),
        nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:

#    引数:
#        z: Encoder Blockへの入力。形状は、(B, N, D)
#        B: バッチサイズ、N:トークンの数、D:ベクトルの長さ
#    返り値:
#        out: Encoder Blockへの出力。形状は、(B, N, D)。[式(10)]
#        B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ()

        # Encoder Blockの前半部分 [式(12)]
        out = self.msa(self.ln1(z)) + z
        # Encoder Blockの後半部分 [式(13)]
        out = self.mlp(self.ln2(out)) + out
        return out
