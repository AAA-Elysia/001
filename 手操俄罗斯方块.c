#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SQUARE_SIZE             20
#define GRID_HORIZONTAL_SIZE    12
#define GRID_VERTICAL_SIZE      20

typedef struct {
    int shape;
    int rotation;
    int x;
    int y;
} CurrentPiece;

CurrentPiece NowPiece;
CurrentPiece NextPiece;

double FallTime = 0.0;
double FallSpeed = 0.6;
double *Speed = &FallSpeed;

int grid[GRID_HORIZONTAL_SIZE][GRID_VERTICAL_SIZE] = {0};
int yulan[5][5] = {0};

// 添加分数变量
int score = 0;

// 方块形状
static const int shapes[7][4][4][2] = {
    // I 型
    {{{0, 0}, {0, 1}, {0, 2}, {0, 3}},
     {{0, 0}, {1, 0}, {2, 0}, {3, 0}},
     {{0, 0}, {0, 1}, {0, 2}, {0, 3}},
     {{0, 0}, {1, 0}, {2, 0}, {3, 0}}},
    // T 型
    {{{0, 0}, {0, 1}, {0, 2}, {1, 1}},
     {{0, 1}, {1, 0}, {1, 1}, {2, 1}},
     {{0, 1}, {1, 0}, {1, 1}, {1, 2}},
     {{0, 1}, {1, 1}, {2, 1}, {1, 0}}},
    // O 型
    {{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
     {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
     {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
     {{0, 0}, {0, 1}, {1, 0}, {1, 1}}},
    // J 型
    {{{0, 0}, {1, 0}, {1, 1}, {1, 2}},
     {{0, 1}, {0, 2}, {1, 1}, {2, 1}},
     {{0, 0}, {0, 1}, {0, 2}, {1, 2}},
     {{0, 1}, {1, 1}, {2, 1}, {2, 0}}},
    // L 型
    {{{0, 2}, {1, 0}, {1, 1}, {1, 2}},
     {{0, 1}, {1, 1}, {2, 1}, {2, 2}},
     {{0, 0}, {0, 1}, {0, 2}, {1, 0}},
     {{0, 0}, {0, 1}, {1, 1}, {2, 1}}},
    // S 型
    {{{0, 1}, {0, 2}, {1, 0}, {1, 1}},
     {{0, 1}, {1, 1}, {1, 2}, {2, 2}},
     {{0, 1}, {0, 2}, {1, 0}, {1, 1}},
     {{0, 1}, {1, 1}, {1, 2}, {2, 2}}},
    // Z 型
    {{{0, 0}, {0, 1}, {1, 1}, {1, 2}},
     {{0, 2}, {1, 1}, {1, 2}, {2, 1}},
     {{0, 0}, {0, 1}, {1, 1}, {1, 2}},
     {{0, 2}, {1, 1}, {1, 2}, {2, 1}}}
};

static const int screenWidth = 500;
static const int screenHeight = 450;

void GetBlock(CurrentPiece *piece) {
    piece->shape = GetRandomValue(0, 6);
    piece->rotation = GetRandomValue(0, 3);
    piece->x = 1;
    piece->y = 1;
}

bool CanMove(CurrentPiece *piece, int x1, int y1) {
    for (int i = 0; i < 4; i++) {
        int New_x = x1 + piece->x + shapes[piece->shape][piece->rotation][i][0];
        int New_y = y1 + piece->y + shapes[piece->shape][piece->rotation][i][1];

        if (New_x < 0 || New_x >= GRID_HORIZONTAL_SIZE) return false;
        if (New_y >= GRID_VERTICAL_SIZE) return false;
        if (New_y >= 0 && grid[New_x][New_y] != 0) return false;
    }
    return true;
}

void PlaceBlock(CurrentPiece *piece) {
    for (int i = 0; i < 4; i++) {
        int x = piece->x + shapes[piece->shape][piece->rotation][i][0];
        int y = piece->y + shapes[piece->shape][piece->rotation][i][1];
        if (y >= 0 && x >= 0 && x < GRID_HORIZONTAL_SIZE) {
            grid[x][y] = 1;
        }
    }
}

// 修改消除行函数，增加分数计算
int ManHang() {
    int linesCleared = 0; // 记录消除的行数
    
    for (int y = GRID_VERTICAL_SIZE - 1; y >= 0; y--) {
        bool FULL = true;
        for (int x = 0; x < GRID_HORIZONTAL_SIZE; x++) {
            if (grid[x][y] == 0) {
                FULL = false;
                break;
            }
        }
        if (FULL) {
            for (int x = 0; x < GRID_HORIZONTAL_SIZE; x++) {
                grid[x][y] = 0;
            }
            for (int y2 = y; y2 > 0; y2--) {
                for (int x = 0; x < GRID_HORIZONTAL_SIZE; x++) {
                    grid[x][y2] = grid[x][y2 - 1];
                }
            }
            y++; // 重新检查当前行
            linesCleared++; // 增加消除行数
        }
    }
    
    // 根据消除的行数计算分数
    switch(linesCleared) {
        case 1:
            score += 100;
            break;
        case 2:
            score += 300;
            break;
        case 3:
            score += 500;
            break;
        case 4:
            score += 800;
            break;
    }
    
    return linesCleared;
}

void YuLanBlock(CurrentPiece *piece) {
    // 清空预览网格
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            yulan[i][j] = 0;
        }
    }

    // 居中绘制下一个方块
    for (int i = 0; i < 4; i++) {
        int x = 2 + shapes[piece->shape][piece->rotation][i][0];
        int y = 2 + shapes[piece->shape][piece->rotation][i][1];
        if (x >= 0 && x < 5 && y >= 0 && y < 5) {
            yulan[x][y] = 1;
        }
    }
}

void DrawGame(CurrentPiece *piece) {
    // 主网格线
    for (int x = 0; x <= GRID_HORIZONTAL_SIZE; x++) {
        DrawLine(x * SQUARE_SIZE, 0, x * SQUARE_SIZE, GRID_VERTICAL_SIZE * SQUARE_SIZE, LIGHTGRAY);
    }
    for (int y = 0; y <= GRID_VERTICAL_SIZE; y++) {
        DrawLine(0, y * SQUARE_SIZE, GRID_HORIZONTAL_SIZE * SQUARE_SIZE, y * SQUARE_SIZE, LIGHTGRAY);
    }

    // 预览区网格线
    for (int x = GRID_HORIZONTAL_SIZE + 5; x <= GRID_HORIZONTAL_SIZE + 10; x++) {
        DrawLine(x * SQUARE_SIZE, 0, x * SQUARE_SIZE, 5 * SQUARE_SIZE, LIGHTGRAY);
    }
    for (int y = 0; y <= 5; y++) {
        DrawLine(GRID_HORIZONTAL_SIZE * SQUARE_SIZE + 5 * SQUARE_SIZE, y * SQUARE_SIZE,
                 GRID_HORIZONTAL_SIZE * SQUARE_SIZE + 10 * SQUARE_SIZE, y * SQUARE_SIZE, LIGHTGRAY);
    }

    // 已落定方块
    for (int x = 0; x < GRID_HORIZONTAL_SIZE; x++) {
        for (int y = 0; y < GRID_VERTICAL_SIZE; y++) {
            if (grid[x][y]) {
                DrawRectangle(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, GRAY);
            }
        }
    }

    // 预览方块
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            if (yulan[x][y]) {
                DrawRectangle((GRID_HORIZONTAL_SIZE + 4 + x) * SQUARE_SIZE, (y - 1) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, GRAY);
            }
        }
    }

    // 当前方块
    for (int i = 0; i < 4; i++) {
        int x = piece->x + shapes[piece->shape][piece->rotation][i][0];
        int y = piece->y + shapes[piece->shape][piece->rotation][i][1];
        DrawRectangle(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, DARKGRAY);
    }
    
    // 显示分数
    DrawText("SCORE:", GRID_HORIZONTAL_SIZE * SQUARE_SIZE + 10, 100, 20, BLACK);
    char scoreText[20];
    sprintf(scoreText, "%d", score);
    DrawText(scoreText, GRID_HORIZONTAL_SIZE * SQUARE_SIZE + 10, 130, 20, BLACK);
    
    // 显示下一个方块标题
    DrawText("NEXT:", GRID_HORIZONTAL_SIZE * SQUARE_SIZE + 10, 10, 20, BLACK);
}

int main(void) {
    InitWindow(screenWidth, screenHeight, "classic game: tetris");
    srand(time(NULL));
    SetTargetFPS(60);

    GetBlock(&NowPiece);
    GetBlock(&NextPiece);
    YuLanBlock(&NextPiece);
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(WHITE);
        FallTime += GetFrameTime();

        if (FallTime >= FallSpeed) {
            if (CanMove(&NowPiece, 0, 1)) {
                NowPiece.y++;
                FallTime = 0.0;
            } else {
                PlaceBlock(&NowPiece);
                ManHang(); // 消除行并更新分数
                // 当前方块落地后，将 NextPiece 赋值给 NowPiece
                NowPiece = NextPiece;
                GetBlock(&NextPiece);
                YuLanBlock(&NextPiece);  // 更新预览区
                if (!CanMove(&NowPiece, 0, 0)) {
                    DrawText("GAME OVER", screenWidth / 2 - 60, screenHeight / 2, 20, RED);
                    EndDrawing();
                    break;
                }
            }
        }

        if (IsKeyPressed(KEY_LEFT) && CanMove(&NowPiece, -1, 0)) {
            NowPiece.x--;
        }
        if (IsKeyPressed(KEY_RIGHT) && CanMove(&NowPiece, 1, 0)) {
            NowPiece.x++;
        }
        if (IsKeyPressed(KEY_UP)) {
            int NewRotation = (NowPiece.rotation + 1) % 4;
            CurrentPiece temp = {NowPiece.shape, NewRotation, NowPiece.x, NowPiece.y};
            if (CanMove(&temp, 0, 0)) {
                NowPiece.rotation = NewRotation;
            }
        }
        if (IsKeyDown(KEY_DOWN) && CanMove(&NowPiece, 0, -1)) {
            *Speed = 0.1;
        } else {
            *Speed = 0.5;
        }

        DrawGame(&NowPiece);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}