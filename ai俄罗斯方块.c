#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define SQUARE_SIZE 20
#define GRID_WIDTH 10
#define GRID_HEIGHT 20

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

typedef struct {
    int shape;
    int rotation;
    int x;
    int y;
} Piece;

int grid[GRID_WIDTH][GRID_HEIGHT] = {0};
Piece currentPiece;
Piece nextPiece;
int score = 0;
double fallTime = 0.0;
double fallSpeed = 0.5;
bool gameOver = false;
bool isAIMode = true;
double aiTimer = 0.0;
double aiSpeed = 0.05;

double weights[5] = {0};
double learningRate = 0.01;
int experienceCount = 0;

// 特征权重索引
#define LINES_CLEARED_WEIGHT 0
#define HEIGHT_WEIGHT 1
#define HOLES_WEIGHT 2
#define BUMPINESS_WEIGHT 3
#define COMPLETE_LINES_WEIGHT 4

void InitGame(void);
void DrawGame(void);
void GetNewPiece(Piece *piece);
bool CanMove(Piece *piece, int dx, int dy);
void PlacePiece(Piece *piece);
void ClearLines(void);
void DropPiece(Piece *piece);
int Holes(int tempGrid[GRID_WIDTH][GRID_HEIGHT]);
int GetHeight(int tempGrid[GRID_WIDTH][GRID_HEIGHT]);
int GetBumpiness(int tempGrid[GRID_WIDTH][GRID_HEIGHT]);
int GetLines(int tempGrid[GRID_WIDTH][GRID_HEIGHT]);
double EvaluateState(int tempGrid[GRID_WIDTH][GRID_HEIGHT]);
void FindBestPlacement(void);
void ResetGame(void);
void ExecuteAction(int action);
double GetStateFeatures(int tempGrid[GRID_WIDTH][GRID_HEIGHT], int featureIndex);
void UpdateWeights(double reward, int linesCleared);

int main(void) {
    // 初始化窗口
    InitWindow(800, 600, "Tetris AI with Machine Learning");
    InitGame();
    SetTargetFPS(60);
    
    // 初始化权重
    weights[LINES_CLEARED_WEIGHT] = 0.760622;
    weights[HEIGHT_WEIGHT] = -0.510066;
    weights[HOLES_WEIGHT] = -0.356630;
    weights[BUMPINESS_WEIGHT] = -0.184483;
    weights[COMPLETE_LINES_WEIGHT] = 0.104606;
    
    // 主游戏循环
    while (!WindowShouldClose()) {
        if (gameOver) {
            if (IsKeyPressed(KEY_R)) {
                ResetGame();
            }
        }
    
        // 切换AI/手动模式
        if (IsKeyPressed(KEY_A)) {
            isAIMode = !isAIMode;
        }
    
        double deltaTime = GetFrameTime();
        fallTime += deltaTime;
        aiTimer += deltaTime;
        
        // AI控制
        if (isAIMode && aiTimer >= aiSpeed) {
            FindBestPlacement();
            aiTimer = 0.0;
        }
        
        // 手动控制
        if (!isAIMode) {
            if (IsKeyPressed(KEY_LEFT) && CanMove(&currentPiece, -1, 0)) {
                currentPiece.x--;
            }
            if (IsKeyPressed(KEY_RIGHT) && CanMove(&currentPiece, 1, 0)) {
                currentPiece.x++;
            }
            if (IsKeyPressed(KEY_UP)) {
                Piece temp = currentPiece;
                temp.rotation = (currentPiece.rotation + 1) % 4;
                if (CanMove(&temp, 0, 0)) {
                    currentPiece.rotation = temp.rotation;
                }
            }
            if (IsKeyDown(KEY_DOWN)) {
                fallSpeed = 0.05;
            } else {
                fallSpeed = 0.5;
            }
        }
        
        // 自动下落
        if (fallTime >= fallSpeed) {
            if (CanMove(&currentPiece, 0, 1)) {
                currentPiece.y++;
                fallTime = 0.0;
            } else {
                // 在放置方块前记录状态
                int prevLines = GetLines(grid);
                int prevHoles = Holes(grid);
                int prevHeight = GetHeight(grid);
                
                PlacePiece(&currentPiece);
                ClearLines();
                
                // 计算奖励
                int newLines = GetLines(grid);
                int newHoles = Holes(grid);
                int newHeight = GetHeight(grid);
                
                int linesCleared = newLines - prevLines;
                int holesDiff = newHoles - prevHoles;
                int heightDiff = newHeight - prevHeight;
                
                double reward = linesCleared * 10.0 - holesDiff * 1.0 - heightDiff * 0.5;
                if (gameOver) reward -= 100; // 游戏结束惩罚
                
                // 更新权重
                UpdateWeights(reward, linesCleared);
                
                currentPiece = nextPiece;
                GetNewPiece(&nextPiece);
                
                // 检查游戏结束
                if (!CanMove(&currentPiece, 0, 0)) {
                    gameOver = true;
                }
                fallTime = 0.0;
            }
        }
        DrawGame();
    }
    
    CloseWindow();
    return 0;
}

void InitGame(void) {
    srand(time(NULL));
    GetNewPiece(&currentPiece);
    GetNewPiece(&nextPiece);
    score = 0;
    fallTime = 0.0;
    gameOver = false;
    experienceCount = 0;
}

void DrawGame(void) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    
    // 绘制游戏区域边框
    DrawRectangleLines(50, 50, GRID_WIDTH * SQUARE_SIZE, GRID_HEIGHT * SQUARE_SIZE, BLACK);
    
    // 绘制网格线
    for (int x = 0; x <= GRID_WIDTH; x++) {
        DrawLine(50 + x * SQUARE_SIZE, 50, 50 + x * SQUARE_SIZE, 50 + GRID_HEIGHT * SQUARE_SIZE, LIGHTGRAY);
    }
    for (int y = 0; y <= GRID_HEIGHT; y++) {
        DrawLine(50, 50 + y * SQUARE_SIZE, 50 + GRID_WIDTH * SQUARE_SIZE, 50 + y * SQUARE_SIZE, LIGHTGRAY);
    }
    
    // 绘制已放置的方块
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            if (grid[x][y]) {
                DrawRectangle(50 + x * SQUARE_SIZE + 1, 50 + y * SQUARE_SIZE + 1, SQUARE_SIZE - 2, SQUARE_SIZE - 2, DARKGRAY);
            }
        }
    }
    
    // 绘制当前方块
    if (!gameOver) {
        for (int i = 0; i < 4; i++) {
            int x = currentPiece.x + shapes[currentPiece.shape][currentPiece.rotation][i][0];
            int y = currentPiece.y + shapes[currentPiece.shape][currentPiece.rotation][i][1];
            if (y >= 0) {
                DrawRectangle(50 + x * SQUARE_SIZE + 1, 50 + y * SQUARE_SIZE + 1, SQUARE_SIZE - 2, SQUARE_SIZE - 2, DARKGRAY);
            }
        }
    }
    
    // 绘制下一个方块预览区域
    DrawText("Next:", 50 + GRID_WIDTH * SQUARE_SIZE + 20, 60, 20, BLACK);
    DrawRectangleLines(50 + GRID_WIDTH * SQUARE_SIZE + 20, 100, 100, 100, BLACK);
    
    // 绘制下一个方块
    for (int i = 0; i < 4; i++) {
        int x = shapes[nextPiece.shape][nextPiece.rotation][i][0];
        int y = shapes[nextPiece.shape][nextPiece.rotation][i][1];
        DrawRectangle(50 + GRID_WIDTH * SQUARE_SIZE + 20 + 40 + x * SQUARE_SIZE,  100 + 40 + y * SQUARE_SIZE + 1, SQUARE_SIZE - 2, SQUARE_SIZE - 2, GRAY);
    }
    
    DrawText(TextFormat("Score: %d", score), 50, 20, 20, BLACK);
    DrawText(TextFormat("Experience: %d", experienceCount), 50, 45, 16, DARKGRAY);
    DrawText(isAIMode ? "AI Mode (Press A to switch)" : "Manual Mode (Press A to switch)", 
             50, 50 + GRID_HEIGHT * SQUARE_SIZE + 20, 16, BLUE);
    
    
    if (gameOver) {
        DrawText("GAME OVER", 150, 200, 30, RED);
        DrawText("Press R to restart", 150, 250, 20, BLACK);
    }
    
    EndDrawing();
}

void GetNewPiece(Piece *piece) {
    piece->shape = GetRandomValue(0, 6);
    piece->rotation = GetRandomValue(0, 3);
    piece->x = GRID_WIDTH / 2 - 1;
    piece->y = 0;
}

bool CanMove(Piece *piece, int dx, int dy) {
    for (int i = 0; i < 4; i++) {
        int newX = piece->x + shapes[piece->shape][piece->rotation][i][0] + dx;
        int newY = piece->y + shapes[piece->shape][piece->rotation][i][1] + dy;
        
        if (newX < 0 || newX >= GRID_WIDTH || newY >= GRID_HEIGHT) {
            return false;
        }
        
        if (newY >= 0 && grid[newX][newY] != 0) {
            return false;
        }
    }
    return true;
}

void PlacePiece(Piece *piece) {
    for (int i = 0; i < 4; i++) {
        int x = piece->x + shapes[piece->shape][piece->rotation][i][0];
        int y = piece->y + shapes[piece->shape][piece->rotation][i][1];
        if (y >= 0) {
            grid[x][y] = piece->shape + 1;
        }
    }
}

void ClearLines(void) {
    int linesClear = 0;
    
    for (int y = GRID_HEIGHT - 1; y >= 0; y--) {
        bool fullLine = true;
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (grid[x][y] == 0) {
                fullLine = false;
                break;
            }
        }
        
        if (fullLine) {
            linesClear++;
            // 下移所有上方的行
            for (int y2 = y; y2 > 0; y2--) {
                for (int x = 0; x < GRID_WIDTH; x++) {
                    grid[x][y2] = grid[x][y2 - 1];
                }
            }
            // 清空顶部行
            for (int x = 0; x < GRID_WIDTH; x++) {
                grid[x][0] = 0;
            }
            y++; // 重新检查当前行
        }
    }
    
    // 更新得分
    switch (linesClear) {
        case 1: score += 100; break;
        case 2: score += 300; break;
        case 3: score += 500; break;
        case 4: score += 800; break;
    }
}

void DropPiece(Piece *piece) {
    while (CanMove(piece, 0, 1)) {
        piece->y++;
    }
}

int Holes(int tempGrid[GRID_WIDTH][GRID_HEIGHT]) {
    int holes = 0;
    for (int x = 0; x < GRID_WIDTH; x++) {
        bool blockFound = false;
        for (int y = 0; y < GRID_HEIGHT; y++) {
            if (tempGrid[x][y] != 0) {
                blockFound = true;
            } else if (blockFound) {
                holes++;
            }
        }
    }
    return holes;
}

int GetHeight(int tempGrid[GRID_WIDTH][GRID_HEIGHT]) {
    int maxHeight = 0;
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            if (tempGrid[x][y] != 0) {
                int height = GRID_HEIGHT - y;
                if (height > maxHeight) {
                    maxHeight = height;
                }
                break;
            }
        }
    }
    return maxHeight;
}

int GetBumpiness(int tempGrid[GRID_WIDTH][GRID_HEIGHT]) {
    int heights[GRID_WIDTH] = {0};
    
    // 计算每列的高度
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            if (tempGrid[x][y] != 0) {
                heights[x] = GRID_HEIGHT - y;
                break;
            }
        }
    }
    
    // 计算不平整度
    int bumpiness = 0;
    for (int i = 0; i < GRID_WIDTH - 1; i++) {
        bumpiness += abs(heights[i] - heights[i + 1]);
    }
    
    return bumpiness;
}

int GetLines(int tempGrid[GRID_WIDTH][GRID_HEIGHT]) {
    int lines = 0;
    for (int y = 0; y < GRID_HEIGHT; y++) {
        bool complete = true;
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (tempGrid[x][y] == 0) {
                complete = false;
                break;
            }
        }
        if (complete) lines++;
    }
    return lines;
}

double GetStateFeatures(int tempGrid[GRID_WIDTH][GRID_HEIGHT], int featureIndex) {
    switch (featureIndex) {
        case LINES_CLEARED_WEIGHT:
            return GetLines(tempGrid);
        case HEIGHT_WEIGHT:
            return GetHeight(tempGrid);
        case HOLES_WEIGHT:
            return Holes(tempGrid);
        case BUMPINESS_WEIGHT:
            return GetBumpiness(tempGrid);
        case COMPLETE_LINES_WEIGHT:
            return GetLines(tempGrid);
        default:
            return 0;
    }
}

double EvaluateState(int tempGrid[GRID_WIDTH][GRID_HEIGHT]) {
    double value = 0;
    for (int i = 0; i < 5; i++) {
        value += weights[i] * GetStateFeatures(tempGrid, i);
    }
    return value;
}

void UpdateWeights(double reward, int linesCleared) {
    // 简单的强化学习更新规则
    if (linesCleared > 0) {
        weights[LINES_CLEARED_WEIGHT] += learningRate * reward;
        weights[HEIGHT_WEIGHT] -= learningRate * 0.1; // 鼓励降低高度
        weights[HOLES_WEIGHT] -= learningRate * 0.1;  // 鼓励减少空洞
    }
    
    // 限制权重范围
    for (int i = 0; i < 5; i++) {
        if (weights[i] > 1.0) weights[i] = 1.0;
        if (weights[i] < -1.0) weights[i] = -1.0;
    }
    
    experienceCount++;
}

void ExecuteAction(int action) {
    switch (action) {
        case 0:
            if (CanMove(&currentPiece, -1, 0)) {
                currentPiece.x--;
            }
            break;
        case 1:
            if (CanMove(&currentPiece, 1, 0)) {
                currentPiece.x++;
            }
            break;
        case 2:
            {
                int newRotation = (currentPiece.rotation + 1) % 4;
                Piece temp = currentPiece;
                temp.rotation = newRotation;
                if (CanMove(&temp, 0, 0)) {
                    currentPiece.rotation = newRotation;
                }
            }
            break;
        case 3:
            DropPiece(&currentPiece);
            break;
    }
}

void FindBestPlacement(void) {
    double bestScore = -999999;
    int bestAction = -1;
    int bestRotation = 0;
    int bestX = currentPiece.x;
    for (int rotation = 0; rotation < 4; rotation++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int tempGrid[GRID_WIDTH][GRID_HEIGHT];
            memcpy(tempGrid, grid, sizeof(grid));
            Piece tempPiece = currentPiece;
            tempPiece.rotation = rotation;
            tempPiece.x = x;
            tempPiece.y = 0;
            if (!CanMove(&tempPiece, 0, 0)) continue;
            while (CanMove(&tempPiece, 0, 1)) {
                tempPiece.y++;
            }
            bool validPlacement = true;
            for (int i = 0; i < 4; i++) {
                int bx = tempPiece.x + shapes[tempPiece.shape][tempPiece.rotation][i][0];
                int by = tempPiece.y + shapes[tempPiece.shape][tempPiece.rotation][i][1];
                if (by >= 0 && bx >= 0 && bx < GRID_WIDTH && by < GRID_HEIGHT) {
                    tempGrid[bx][by] = tempPiece.shape + 1;
                } else {
                    validPlacement = false;
                    break;
                }
            }
            
            if (validPlacement) {
                double score = EvaluateState(tempGrid);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestRotation = rotation;
                    bestX = x;
                }
            }
        }
    }
    if (bestRotation != currentPiece.rotation) {
        int newRotation = bestRotation;
        Piece temp = currentPiece;
        temp.rotation = newRotation;
        if (CanMove(&temp, 0, 0)) {
            currentPiece.rotation = newRotation;
            return;
        }
    }
    
    if (bestX < currentPiece.x) {
        if (CanMove(&currentPiece, -1, 0)) {
            currentPiece.x--;
            return;
        }
    } else if (bestX > currentPiece.x) {
        if (CanMove(&currentPiece, 1, 0)) {
            currentPiece.x++;
            return;
        }
    } else {
        DropPiece(&currentPiece);
    }
}

void ResetGame(void) {
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            grid[x][y] = 0;
        }
    }
    
    GetNewPiece(&currentPiece);
    GetNewPiece(&nextPiece);
    score = 0;
    fallTime = 0.0;
    gameOver = false;
    experienceCount = 0;
}