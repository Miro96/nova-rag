#include <stdio.h>
#include "utils.h"

struct User {
    int id;
    char name[100];
};

int get_user(int id) {
    printf("Getting user %d\n", id);
    return validate_id(id);
}

int validate_id(int id) {
    return id > 0;
}
